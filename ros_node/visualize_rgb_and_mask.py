"""
Renders a side-by-side (RGB | binary mask) MP4 per already-produced
mask_extraction_output/*_masks.bag: left panel is the actual RGB topic that
was processed (the throttled /oak/image_raw/throttle for 14/15 bags, or the
native /oak/image_raw for the one bag that only has that), right panel is
/mask_extraction/body_mask scaled from {0,1} to {0,255} for visibility. Both
topics already live in the same output bag (the RGB is an untouched copy of
the original, the mask carries the same header.stamp), so no input bag or
pipeline re-run is needed -- just paired reads by message order (confirmed
the timestamps already line up exactly).

Usage:
    source /opt/ros/noetic/setup.bash
    pyenv activate gdino310
    python3 visualize_rgb_and_mask.py --output-dir ./mask_extraction_output
"""
import os
import sys
import glob
import argparse

import numpy as np
import cv2
import rosbag

IMAGE_TOPIC_CANDIDATES = ["/oak/image_raw", "/oak/image_raw/throttle"]
MASK_TOPIC = "/mask_extraction/upper_body_mask"
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


def decode_binary_mask(msg):
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
    vis = (arr * 255).astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


def measure_hz(stamps):
    if len(stamps) < 2:
        return 0.0
    diffs = np.diff(stamps)
    diffs = diffs[diffs > 0]
    return 1.0 / np.mean(diffs) if len(diffs) else 0.0


def render_bag(bag_path, out_video_path, image_topic):
    with rosbag.Bag(bag_path) as bag:
        rgb_msgs = [msg for _, msg, _ in bag.read_messages(topics=[image_topic])]
        mask_msgs = [msg for _, msg, _ in bag.read_messages(topics=[MASK_TOPIC])]

    n = min(len(rgb_msgs), len(mask_msgs))
    if n == 0:
        print(f"  SKIP: no paired frames ({len(rgb_msgs)} rgb, {len(mask_msgs)} mask)")
        return 0

    stamps = [m.header.stamp.to_sec() for m in rgb_msgs[:n]]
    hz = measure_hz(stamps)

    video_writer = None
    for i in range(n):
        rgb = decode_image(rgb_msgs[i])
        mask_vis = decode_binary_mask(mask_msgs[i])
        combined = np.hstack([rgb, mask_vis])
        cv2.putText(combined, "RGB (throttled)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, "uper_body_mask", (rgb.shape[1] + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if video_writer is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
            write_fps = hz if hz > 0 else 5.0
            video_writer = cv2.VideoWriter(out_video_path, fourcc, write_fps, (w, h))
        video_writer.write(combined)

    video_writer.release()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./mask_extraction_output",
                        help="Directory containing the *_masks.bag files (output goes here too).")
    parser.add_argument("--only", default=None,
                        help="Substring filter to process a subset of bag filenames.")
    args = parser.parse_args()

    bag_paths = sorted(glob.glob(os.path.join(args.output_dir, "*_masks.bag")))
    if args.only:
        bag_paths = [p for p in bag_paths if args.only in os.path.basename(p)]
    if not bag_paths:
        print(f"No *_masks.bag files found in {args.output_dir}")
        sys.exit(1)

    for i, bag_path in enumerate(bag_paths, 1):
        stem = os.path.splitext(os.path.basename(bag_path))[0].removesuffix("_masks")
        out_video_path = os.path.join(args.output_dir, f"{stem}_rgb_and_garment_mask.mp4")
        image_topic = detect_image_topic(bag_path)
        print(f"[{i}/{len(bag_paths)}] {os.path.basename(bag_path)} (topic: {image_topic})")
        if image_topic is None:
            print(f"  SKIP: no RGB topic found (checked {IMAGE_TOPIC_CANDIDATES})")
            continue
        n = render_bag(bag_path, out_video_path, image_topic)
        if n:
            print(f"  wrote {out_video_path} ({n} frames)")


if __name__ == "__main__":
    main()
