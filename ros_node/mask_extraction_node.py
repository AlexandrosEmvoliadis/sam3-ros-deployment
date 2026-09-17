import os
import sys
import json
import time
import argparse
from collections import deque

import numpy as np
import cv2
import torch
import tensorrt as trt
from PIL import Image

import rospy
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String

from transformers.models.sam3 import Sam3Processor
from transformers import AutoTokenizer

from body_segmentation import filter_and_rank_people, to_bool_mask

SAM3_MODEL_ID = "facebook/sam3"
ENCODER_TRT = "./onnx_weights/sam3_encoder_fp16.engine"
DECODER_BATCH_TRT = "./onnx_weights/sam3_decoder_batched_fp16.engine"

IMAGE_TOPIC = "/oak/image_raw/throttle"
OUTPUT_IMAGE_TOPIC = "/mask_extraction/image"
OUTPUT_RESULTS_TOPIC = "/mask_extraction/results"


UPPER_BODY_QUERIES = ["head and upper body"]
HEAD_QUERIES = ["head"]
UPPER_BODY_ONLY_QUERIES = ["upper garment"] #SAM-3 cannot segment upper body as is, so we use the top garment one should wear
QUERIES = ["Hard Hat"] + UPPER_BODY_QUERIES + HEAD_QUERIES + UPPER_BODY_ONLY_QUERIES
DETECTION_THRESHOLD = 0.6
MASK_THRESHOLD = 0.5

HELMET_MIN_OVERLAP = 0.60 #threshold to decide if a detected, not-filtered person wears the hat

FPS_LOG_EVERY = 10
VIDEO_FOURCC = "mp4v"
VIDEO_FPS_FALLBACK = 5.0  # used for the writer only until the source rate is measured


def _build_output_obj(raw_dict):
    class _Out:
        pass
    out = _Out()
    for name in ["pred_masks", "pred_logits", "pred_boxes",
                "semantic_seg", "presence_logits", "decoder_reference_boxes"]:
        v = raw_dict.get(name)
        if v is None:
            setattr(out, name, None)
        elif isinstance(v, np.ndarray):
            setattr(out, name, torch.from_numpy(v))
        else:
            setattr(out, name, v)
    return out


class Sam3TrtBatchedPipeline:
    def __init__(self, encoder_path, decoder_path, queries):
        self.processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(SAM3_MODEL_ID)
        self.queries = queries
        self.n_queries = len(queries)

        self.torch_device = torch.device("cuda:0")
        torch.cuda.set_device(self.torch_device)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        def load_engine(path):
            with open(path, "rb") as f:
                return runtime.deserialize_cuda_engine(f.read())

        self.enc_engine = load_engine(encoder_path)
        self.dec_engine = load_engine(decoder_path)
        self.enc_ctx = self.enc_engine.create_execution_context()
        self.dec_ctx = self.dec_engine.create_execution_context()
        self.enc_io = self._get_io(self.enc_engine)
        self.dec_io = self._get_io(self.dec_engine)

        tok = tokenizer(queries, return_tensors="pt", padding=True)
        self.input_ids = tok["input_ids"].to(self.torch_device, dtype=torch.int64).contiguous()
        self.attention_mask = tok["attention_mask"].to(self.torch_device, dtype=torch.int64).contiguous()

        self._warmed_up = False

    @staticmethod
    def _get_io(engine):
        info = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            info[name] = {"is_input": mode == trt.TensorIOMode.INPUT}
        return info

    def _run(self, ctx, io_info, feed_gpu):
        tensors = {}
        for name, info in io_info.items():
            if info["is_input"]:
                t = feed_gpu[name].contiguous()
                ctx.set_input_shape(name, tuple(t.shape))
                tensors[name] = t
                ctx.set_tensor_address(name, t.data_ptr())
        for name, info in io_info.items():
            if not info["is_input"]:
                shape = ctx.get_tensor_shape(name)
                t = torch.empty(tuple(shape), dtype=torch.float32, device=self.torch_device)
                tensors[name] = t
                ctx.set_tensor_address(name, t.data_ptr())
        ctx.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return {n: tensors[n] for n, info in io_info.items() if not info["is_input"]}

    def _infer_raw(self, pv_gpu):
        enc_out = self._run(self.enc_ctx, self.enc_io, {"pixel_values": pv_gpu})

        bat_feed = {}
        for n in self.dec_io:
            if self.dec_io[n]["is_input"] and n in enc_out:
                bat_feed[n] = enc_out[n].expand(self.n_queries, *enc_out[n].shape[1:]).contiguous()
        bat_feed["input_ids"] = self.input_ids
        bat_feed["attention_mask"] = self.attention_mask

        return self._run(self.dec_ctx, self.dec_io, bat_feed)

    def warmup(self, image, runs=3):
        pv = self.processor(images=image, text=self.queries[0], return_tensors="pt")["pixel_values"]
        pv_gpu = pv.to(self.torch_device).contiguous()
        for _ in range(runs):
            self._infer_raw(pv_gpu)
        self._warmed_up = True

    def segment(self, image):
        """Returns (masks_by_query, scores_by_query) for self.queries."""
        inputs = self.processor(images=image, text=self.queries[0], return_tensors="pt")
        pv_gpu = inputs["pixel_values"].to(self.torch_device).contiguous()
        original_sizes = inputs.get("original_sizes")
        original_sizes = original_sizes.tolist() if original_sizes is not None \
            else [[image.size[1], image.size[0]]]

        if not self._warmed_up:
            self.warmup(image)

        dec_out = self._infer_raw(pv_gpu)

        masks_by_query, scores_by_query = {}, {}
        for qi, q in enumerate(self.queries):
            raw_i = {n: v[qi:qi + 1].cpu() for n, v in dec_out.items()}
            out = _build_output_obj(raw_i)
            results = self.processor.post_process_instance_segmentation(
                out, threshold=DETECTION_THRESHOLD, mask_threshold=MASK_THRESHOLD,
                target_sizes=original_sizes,
            )[0]
            masks = results["masks"]
            scores_tensor = results.get("scores")
            if scores_tensor is None or len(scores_tensor) != len(masks):
                scores_list = [0.0] * len(masks)
            else:
                scores_list = scores_tensor.cpu().tolist()
            masks_by_query[q] = [m.cpu() for m in masks]
            scores_by_query[q] = scores_list

        return masks_by_query, scores_by_query

def check_ppe_overlap(region_mask_bin, item_masks, min_overlap):
    if region_mask_bin is None or region_mask_bin.sum() == 0:
        return False, 0.0

    best_overlap = 0.0
    for item_mask in item_masks:
        item_bin = to_bool_mask(item_mask)
        area = item_bin.sum()
        if area == 0:
            continue
        overlap = float(np.logical_and(item_bin, region_mask_bin).sum()) / float(area)
        best_overlap = max(best_overlap, overlap)

    return best_overlap >= min_overlap, best_overlap


def helmet_verdict(cand, hardhat_masks, min_overlap=HELMET_MIN_OVERLAP):
    if not cand.get("head_visible", True):
        return "unknown", 0.0
    region = cand["head_mask_bin"] | cand["mask_bin"]
    worn, overlap = check_ppe_overlap(region, hardhat_masks, min_overlap)
    return ("worn" if worn else "not_worn"), overlap


VERDICT_COLORS = {
    "worn": (0, 200, 0),        # green, BGR
    "not_worn": (0, 0, 220),    # red
    "unknown": (0, 200, 220),   # yellow -- head/face not visible, not "confirmed bare"
}
VERDICT_LABELS = {
    "worn": "Helmet OK",
    "not_worn": "NO HELMET",
    "unknown": "HEAD NOT VISIBLE",
}


def draw_frame(bgr_frame, candidates, helmet_results, achieved_fps, source_hz):
    overlay = bgr_frame.copy()

    for cand, (verdict, _overlap) in zip(candidates, helmet_results):
        m8 = cand["mask_bin"].astype(np.uint8) * 255
        roi = m8 > 127
        color = np.array(VERDICT_COLORS[verdict])
        overlay[roi] = (overlay[roi] * 0.6 + color * 0.4).astype(np.uint8)

    vis = overlay
    for cand, (verdict, overlap) in zip(candidates, helmet_results):
        ys, xs = np.where(cand["mask_bin"])
        if len(xs) == 0:
            continue
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        color = VERDICT_COLORS[verdict]
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = VERDICT_LABELS[verdict] if verdict == "unknown" \
            else f"{VERDICT_LABELS[verdict]} ({overlap:.0%})"
        cv2.putText(vis, label, (box[0], max(15, box[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    status = f"pipeline: {achieved_fps:.1f} FPS | source: {source_hz:.1f} Hz"
    cv2.putText(vis, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vis

class FpsTracker:
    def __init__(self, window=30):
        self.proc_times = deque(maxlen=window)
        self.msg_stamps = deque(maxlen=window)

    def on_message(self, stamp_sec):
        self.msg_stamps.append(stamp_sec)

    def record_processing(self, dt):
        self.proc_times.append(dt)

    @property
    def achieved_fps(self):
        if not self.proc_times:
            return 0.0
        return 1.0 / np.mean(self.proc_times)

    @property
    def source_hz(self):
        if len(self.msg_stamps) < 2:
            return 0.0
        diffs = np.diff(self.msg_stamps)
        diffs = diffs[diffs > 0]
        return 1.0 / np.mean(diffs) if len(diffs) else 0.0

class MaskExtractionNode:
    def __init__(self, image_topic, output_video_path):
        rospy.loginfo("Loading SAM-3 TensorRT pipeline...")
        self.pipeline = Sam3TrtBatchedPipeline(ENCODER_TRT, DECODER_BATCH_TRT, QUERIES)
        self.fps = FpsTracker()
        self.frame_count = 0
        self.video_writer = None
        self.output_video_path = output_video_path

        self.image_pub = rospy.Publisher(OUTPUT_IMAGE_TOPIC, RosImage, queue_size=1)
        self.results_pub = rospy.Publisher(OUTPUT_RESULTS_TOPIC, String, queue_size=1)
        self.sub = rospy.Subscriber(image_topic, RosImage, self.callback,
                                    queue_size=1, buff_size=2**24)

        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(f"mask_extraction node ready, subscribed to {image_topic}")

    def decode_image(self, msg):
        if msg.encoding == "mono8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if msg.encoding != "bgr8":
            rospy.logwarn_once(f"Unhandled encoding '{msg.encoding}', assuming bgr8 layout")
        return arr

    def callback(self, msg):
        t0 = time.perf_counter()
        self.fps.on_message(msg.header.stamp.to_sec())

        bgr = self.decode_image(msg)
        rgb_image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        masks_by_query, scores_by_query = self.pipeline.segment(rgb_image)
        upper_body_masks = [m for q in UPPER_BODY_QUERIES for m in masks_by_query.get(q, [])]
        upper_body_scores = [s for q in UPPER_BODY_QUERIES for s in scores_by_query.get(q, [])]
        head_masks = [m for q in HEAD_QUERIES for m in masks_by_query.get(q, [])]
        hardhat_masks = masks_by_query.get("Hard Hat", [])

        image_area = bgr.shape[0] * bgr.shape[1]
        candidates, _diagnostics = filter_and_rank_people(
            upper_body_masks, head_masks, image_area, upper_body_scores=upper_body_scores)

        helmet_results = [helmet_verdict(c, hardhat_masks) for c in candidates]

        dt = time.perf_counter() - t0
        self.fps.record_processing(dt)
        self.frame_count += 1

        vis = draw_frame(bgr, candidates, helmet_results, self.fps.achieved_fps, self.fps.source_hz)
        self.write_video_frame(vis)
        self.publish_image(vis, msg.header)
        self.publish_results(candidates, helmet_results, msg.header, dt)

        if self.frame_count % FPS_LOG_EVERY == 0:
            keeping_up = "OK" if self.fps.achieved_fps >= self.fps.source_hz else "FALLING BEHIND"
            rospy.loginfo(f"[frame {self.frame_count}] {len(candidates)} people kept | "
                          f"pipeline {self.fps.achieved_fps:.2f} FPS vs source "
                          f"{self.fps.source_hz:.2f} Hz -> {keeping_up}")

    def write_video_frame(self, vis):
        if self.video_writer is None:
            h, w = vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
            write_fps = self.fps.source_hz if self.fps.source_hz > 0 else VIDEO_FPS_FALLBACK
            self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, write_fps, (w, h))
            rospy.loginfo(f"Writing annotated video to {self.output_video_path} at {write_fps:.1f} FPS")
        self.video_writer.write(vis)

    def publish_image(self, bgr, header):
        out_msg = RosImage()
        out_msg.header = header
        out_msg.height, out_msg.width = bgr.shape[:2]
        out_msg.encoding = "bgr8"
        out_msg.is_bigendian = 0
        out_msg.step = bgr.shape[1] * 3
        out_msg.data = bgr.tobytes()
        self.image_pub.publish(out_msg)

    def publish_results(self, candidates, helmet_results, header, dt):
        people = []
        for cand, (verdict, overlap) in zip(candidates, helmet_results):
            ys, xs = np.where(cand["mask_bin"])
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
            people.append({
                "box": box, "score": cand["score"], "area_frac": cand["area_frac"],
                "helmet_verdict": verdict, "helmet_overlap": round(float(overlap), 3),
            })
        payload = {
            "stamp": header.stamp.to_sec(), "frame": self.frame_count,
            "processing_time_ms": round(dt * 1000, 1), "people": people,
        }
        self.results_pub.publish(String(data=json.dumps(payload)))

    def shutdown(self):
        if self.video_writer is not None:
            self.video_writer.release()
            rospy.loginfo(f"Saved annotated video: {self.output_video_path} "
                          f"({self.frame_count} frames)")
        rospy.loginfo(f"Final: pipeline {self.fps.achieved_fps:.2f} FPS vs "
                      f"source {self.fps.source_hz:.2f} Hz over {self.frame_count} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default=IMAGE_TOPIC)
    parser.add_argument("--output-video", default=None)
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    output_video = args.output_video or f"./mask_extraction_output/run_{int(time.time())}.mp4"
    out_dir = os.path.dirname(output_video) or "."
    os.makedirs(out_dir, exist_ok=True)

    rospy.init_node("mask_extraction", anonymous=False)
    node = MaskExtractionNode(args.topic, output_video)
    rospy.spin()
