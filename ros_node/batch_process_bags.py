"""
Offline bag-to-bag mask extraction: for each input .bag, copies every original
message through unchanged into a new output .bag, and for each OAK camera
frame additionally writes 4 new topics carrying the extracted results:

  /mask_extraction/image            sensor_msgs/Image  (bgr8, annotated)
  /mask_extraction/results          std_msgs/String     (JSON per-frame)
  /mask_extraction/body_mask        sensor_msgs/Image  (mono8, binary: 1
                                     where any kept person's "head and
                                     upper body" region is, 0 elsewhere --
                                     union across all kept people, not
                                     per-person indexed)
  /mask_extraction/upper_body_mask  sensor_msgs/Image  (mono8, binary: 1
                                     where any kept person's separate
                                     torso-only "upper body" region is, 0
                                     elsewhere -- for downstream LiDAR
                                     fusion, e.g. checking jacket/hi-vis
                                     vest wear against a region that
                                     doesn't also claim head pixels)

/mask_extraction/results carries the per-person detail (box/score/helmet
verdict) that the binary masks can't -- they're plain foreground
indicators, not label maps, so joining a specific mask blob back to a
specific person's metadata isn't possible from the masks alone.

Also compiles a per-bag MP4 of the annotated frames, at the bag's own image
topic rate, and reports achieved pipeline FPS vs. that rate.

Reuses the exact same TensorRT SAM-3 pipeline, filtering, and helmet rule as
mask_extraction_node.py (the live rospy node) -- this script just drives them
by reading bag messages directly instead of a live subscription, since a
full-bag-in/bag-out conversion is a deterministic one-shot job.

Usage:
    pyenv activate gdino310
    python3 batch_process_bags.py --bags-dir /data2/PPE/bags \
        --output-dir ./mask_extraction_output
"""
import os
import sys
import json
import time
import glob
import argparse

import numpy as np
import cv2
import rosbag
from PIL import Image as PILImage
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage

from mask_extraction_node import (
    Sam3TrtBatchedPipeline, helmet_verdict, ENCODER_TRT, DECODER_BATCH_TRT,
    QUERIES, UPPER_BODY_QUERIES, HEAD_QUERIES, UPPER_BODY_ONLY_QUERIES, HELMET_MIN_OVERLAP,
    VERDICT_COLORS, VERDICT_LABELS,
)
from body_segmentation import filter_and_rank_people

IMAGE_TOPIC_CANDIDATES = ["/oak/image_raw", "/oak/image_raw/throttle"]
VIDEO_FOURCC = "mp4v"


def detect_image_topic(bag_path, candidates=IMAGE_TOPIC_CANDIDATES):
    with rosbag.Bag(bag_path) as bag:
        info = bag.get_type_and_topic_info().topics
    for candidate in candidates:
        if candidate in info and info[candidate].message_count > 0:
            return candidate
    return None


def decode_image(msg):
    if msg.encoding == "mono8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def make_image_msg(bgr, header, encoding="bgr8"):
    msg = RosImage()
    msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = bgr.shape[1] * (1 if encoding == "mono8" else 3)
    msg.data = bgr.tobytes()
    return msg


def draw_frame(bgr_frame, candidates, helmet_results):
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
    return vis


def build_binary_mask(shape_hw, candidates, mask_key="mask_bin"):
    mask = np.zeros(shape_hw, dtype=np.uint8)
    for cand in candidates:
        mask[cand[mask_key]] = 1
    return mask


def measure_source_hz(bag_path, topic):
    stamps = []
    with rosbag.Bag(bag_path) as bag:
        for _, msg, _t in bag.read_messages(topics=[topic]):
            stamps.append(msg.header.stamp.to_sec())
    if len(stamps) < 2:
        return 0.0, len(stamps)
    diffs = np.diff(stamps)
    diffs = diffs[diffs > 0]
    return (1.0 / np.mean(diffs) if len(diffs) else 0.0), len(stamps)


def process_bag(pipeline, in_path, out_bag_path, out_video_path, image_topic):
    source_hz, n_frames_expected = measure_source_hz(in_path, image_topic)
    print(f"  source rate: {source_hz:.2f} Hz over {n_frames_expected} frames")

    video_writer = None
    frame_count = 0
    proc_times = []

    with rosbag.Bag(in_path) as in_bag, \
         rosbag.Bag(out_bag_path, "w", compression=rosbag.Compression.LZ4) as out_bag:

        for topic, msg, t in in_bag.read_messages():
            out_bag.write(topic, msg, t)

            if topic != image_topic:
                continue

            t0 = time.perf_counter()
            bgr = decode_image(msg)
            rgb_image_pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            masks_by_query, scores_by_query = pipeline.segment(rgb_image_pil)
            upper_body_masks = [m for q in UPPER_BODY_QUERIES for m in masks_by_query.get(q, [])]
            upper_body_scores = [s for q in UPPER_BODY_QUERIES for s in scores_by_query.get(q, [])]
            head_masks = [m for q in HEAD_QUERIES for m in masks_by_query.get(q, [])]
            upper_body_only_masks = [m for q in UPPER_BODY_ONLY_QUERIES for m in masks_by_query.get(q, [])]
            hardhat_masks = masks_by_query.get("Hard Hat", [])

            image_area = bgr.shape[0] * bgr.shape[1]
            candidates, _diag = filter_and_rank_people(
                upper_body_masks, head_masks, image_area, upper_body_scores=upper_body_scores,
                upper_body_only_masks=upper_body_only_masks)
            helmet_results = [helmet_verdict(c, hardhat_masks, HELMET_MIN_OVERLAP)
                              for c in candidates]

            dt = time.perf_counter() - t0
            proc_times.append(dt)
            frame_count += 1

            vis = draw_frame(bgr, candidates, helmet_results)
            status = f"pipeline: {1.0/np.mean(proc_times[-30:]):.1f} FPS | source: {source_hz:.1f} Hz"
            cv2.putText(vis, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            if video_writer is None:
                h, w = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
                write_fps = source_hz if source_hz > 0 else 5.0
                video_writer = cv2.VideoWriter(out_video_path, fourcc, write_fps, (w, h))
            video_writer.write(vis)

            body_mask = build_binary_mask(bgr.shape[:2], candidates, "mask_bin")
            upper_body_mask = build_binary_mask(bgr.shape[:2], candidates, "upper_body_only_mask_bin")

            out_bag.write("/mask_extraction/image", make_image_msg(vis, msg.header), t)
            out_bag.write("/mask_extraction/body_mask",
                          make_image_msg(body_mask, msg.header, "mono8"), t)
            out_bag.write("/mask_extraction/upper_body_mask",
                          make_image_msg(upper_body_mask, msg.header, "mono8"), t)

            people = []
            for cand, (verdict, overlap) in zip(candidates, helmet_results):
                ys, xs = np.where(cand["mask_bin"])
                box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
                people.append({
                    "box": box, "score": cand["score"],
                    "area_frac": cand["area_frac"], "helmet_verdict": verdict,
                    "helmet_overlap": round(float(overlap), 3),
                })
            payload = {
                "stamp": msg.header.stamp.to_sec(), "frame": frame_count,
                "processing_time_ms": round(dt * 1000, 1), "people": people,
            }
            out_bag.write("/mask_extraction/results", String(data=json.dumps(payload)), t)

    if video_writer is not None:
        video_writer.release()

    achieved_fps = 1.0 / np.mean(proc_times) if proc_times else 0.0
    print(f"  processed {frame_count} frames | pipeline {achieved_fps:.2f} FPS vs source {source_hz:.2f} Hz "
          f"-> {'OK' if achieved_fps >= source_hz else 'FALLING BEHIND'}")
    print(f"  wrote {out_bag_path}")
    print(f"  wrote {out_video_path}")
    return frame_count, achieved_fps, source_hz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bags-dir", default="/data2/PPE/bags")
    parser.add_argument("--output-dir", default="./mask_extraction_output")
    parser.add_argument("--topic", default=None,
                        help="Force a specific image topic; default auto-detects per bag.")
    parser.add_argument("--only", default=None,
                        help="Substring filter to process a subset of bag filenames.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bag_paths = sorted(glob.glob(os.path.join(args.bags_dir, "*.bag")))
    if args.only:
        bag_paths = [p for p in bag_paths if args.only in os.path.basename(p)]
    if not bag_paths:
        print(f"No .bag files found in {args.bags_dir}")
        sys.exit(1)

    print(f"Loading SAM-3 TensorRT pipeline ({QUERIES})...")
    pipeline = Sam3TrtBatchedPipeline(ENCODER_TRT, DECODER_BATCH_TRT, QUERIES)

    summary = []
    for i, bag_path in enumerate(bag_paths, 1):
        stem = os.path.splitext(os.path.basename(bag_path))[0]
        out_bag_path = os.path.join(args.output_dir, f"{stem}_masks.bag")
        out_video_path = os.path.join(args.output_dir, f"{stem}.mp4")

        image_topic = args.topic or detect_image_topic(bag_path)
        print(f"\n[{i}/{len(bag_paths)}] {os.path.basename(bag_path)} (topic: {image_topic})")
        if image_topic is None:
            print(f"  SKIP: no OAK image topic found (checked {IMAGE_TOPIC_CANDIDATES})")
            summary.append((stem, 0, 0.0, 0.0))
            continue

        n_frames, achieved_fps, source_hz = process_bag(
            pipeline, bag_path, out_bag_path, out_video_path, image_topic)
        summary.append((stem, n_frames, achieved_fps, source_hz))

    print("\n" + "=" * 70)
    print(f"{'Bag':<40} {'Frames':>7} {'Pipeline FPS':>13} {'Source Hz':>10}")
    print("-" * 70)
    for stem, n_frames, achieved_fps, source_hz in summary:
        print(f"{stem:<40} {n_frames:>7} {achieved_fps:>13.2f} {source_hz:>10.2f}")


if __name__ == "__main__":
    main()
