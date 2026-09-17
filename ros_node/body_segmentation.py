import os

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from transformers.models.sam3 import Sam3Processor, Sam3Model
from transformers import AutoTokenizer

SAM3_MODEL_ID = "facebook/sam3"
IMAGE_FOLDER = "/data2/PPE/ppe_thessaloniki_extracted_images"
OUTPUT_DIR = "./upper_body_test"
DETECTION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

MAX_PERSONS_PER_IMAGE = None  # e.g. set to 1 to keep only the largest/closest candidate

# See module docstring: fixed floor placed between confirmed reflection
# artifacts (area_frac 0.00053-0.00062) and a confirmed real small/distant
# person (0.00093).
MIN_AREA_FRAC = 0.0008

# A person whose own mask touches the top edge (camera too close, head
# cropped out of the frame) has no visible head at all -- distinct from a
# genuinely bare head, which is real information. Used to mark head_visible
# so the helmet verdict can be reported as "unknown", not "not worn".
HEAD_CROP_EDGE_MARGIN = 5

device = "cuda"

# The upper-body region comes directly from a SAM-3 "head and upper body"
# prompt (a real segmentation of the visible torso/shoulders/head) and is
# used as the anchor for each person -- see match_person_to_upper_body,
# reused generically for matching "head" to that same anchor. No minimum
# overlap required (no filtering) -- any positive overlap counts as a
# match.
MIN_UPPER_BODY_MATCH_OVERLAP = 0.0


def is_valid_upper_body(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask_bin = mask > 0.5
    total_pixels = mask_bin.sum()
    if total_pixels == 0:
        return False, {"reason": "empty", "pixels": 0}

    ys, xs = np.where(mask_bin)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    mask_h = y_max - y_min
    mask_w = x_max - x_min

    return True, {
        "reason": "upper body detected",
        "pixels": int(total_pixels),
        "aspect_ratio": float(mask_h / max(mask_w, 1)),
        "y_min": int(y_min), "y_max": int(y_max),
        "x_min": int(x_min), "x_max": int(x_max),
        "mask_h": int(mask_h), "mask_w": int(mask_w),
    }


def mask_to_box(mask_bin):
    ys, xs = np.where(mask_bin)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

MATCH_TIE_TOLERANCE = 0.03


def match_person_to_upper_body(person_box, upper_body_boxes,
                                min_overlap=MIN_UPPER_BODY_MATCH_OVERLAP):
    overlaps = []
    for ub_box in upper_body_boxes:
        overlaps.append(bbox_overlap_ratio(person_box, ub_box) if ub_box is not None else None)
    valid = [i for i, o in enumerate(overlaps) if o is not None]
    if not valid:
        return None, 0.0
    max_overlap = max(overlaps[i] for i in valid)
    if max_overlap < min_overlap:
        return None, max_overlap

    def tie_key(i):
        b = upper_body_boxes[i]
        return (b[2] - b[0]) * (b[3] - b[1])

    tied = [i for i in valid if overlaps[i] >= max_overlap - MATCH_TIE_TOLERANCE]
    best_idx = max(tied, key=tie_key)
    return best_idx, overlaps[best_idx]


def to_bool_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    return mask > 0.5


def mask_area_frac(mask, image_area):
    return float(to_bool_mask(mask).sum()) / image_area


def bbox_overlap_ratio(box_a, box_b):
    ax_min, ay_min, ax_max, ay_max = box_a
    bx_min, by_min, bx_max, by_max = box_b

    ix_min, iy_min = max(ax_min, bx_min), max(ay_min, by_min)
    ix_max, iy_max = min(ax_max, bx_max), min(ay_max, by_max)
    if ix_max <= ix_min or iy_max <= iy_min:
        return 0.0
    intersection = (ix_max - ix_min) * (iy_max - iy_min)

    area_a = (ax_max - ax_min) * (ay_max - ay_min)
    area_b = (bx_max - bx_min) * (by_max - by_min)
    smaller_area = min(area_a, area_b)
    return float(intersection) / float(smaller_area) if smaller_area > 0 else 0.0

def filter_and_rank_people(upper_body_masks, head_masks, image_area,
                           upper_body_scores=None, upper_body_only_masks=None):
    diagnostics = []
    candidates = []

    upper_body_only_masks = upper_body_only_masks or []
    upper_body_bins = [to_bool_mask(m) for m in upper_body_masks]
    head_bins = [to_bool_mask(m) for m in head_masks]
    head_boxes = [mask_to_box(b) for b in head_bins]
    upper_body_only_bins = [to_bool_mask(m) for m in upper_body_only_masks]
    upper_body_only_boxes = [mask_to_box(b) for b in upper_body_only_bins]

    for i, mask in enumerate(upper_body_masks):
        score = float(upper_body_scores[i]) if upper_body_scores is not None else None
        entry = {"index": i, "score": score}
        mask_bin = upper_body_bins[i]

        is_valid, info = is_valid_upper_body(mask)
        if not is_valid:
            entry.update(status="rejected", reason=info["reason"])
            diagnostics.append(entry)
            continue

        area_frac = info["pixels"] / image_area
        if area_frac < MIN_AREA_FRAC:
            entry.update(status="rejected",
                         reason=f"too small, likely a reflection (frac={area_frac:.5f})")
            diagnostics.append(entry)
            continue

        anchor_box = (info["x_min"], info["y_min"], info["x_max"], info["y_max"])
        head_idx, head_overlap = match_person_to_upper_body(anchor_box, head_boxes)
        if head_idx is not None:
            head_mask_bin = head_bins[head_idx]
            head_visible = head_boxes[head_idx][1] >= HEAD_CROP_EDGE_MARGIN
        else:
            head_mask_bin = np.zeros_like(mask_bin, dtype=bool)
            head_visible = False
        ub_only_idx, ub_only_overlap = match_person_to_upper_body(anchor_box, upper_body_only_boxes)
        upper_body_only_mask_bin = upper_body_only_bins[ub_only_idx] if ub_only_idx is not None \
            else np.zeros_like(mask_bin, dtype=bool)

        box = anchor_box
        entry.update(status="kept", mask=mask, mask_bin=mask_bin, info=info, box=box,
                    area_frac=area_frac, head_mask_bin=head_mask_bin,
                    head_index=head_idx, head_match_overlap=head_overlap,
                    head_visible=head_visible, upper_body_only_mask_bin=upper_body_only_mask_bin,
                    upper_body_only_index=ub_only_idx, upper_body_only_match_overlap=ub_only_overlap)
        diagnostics.append(entry)
        candidates.append(entry)

    candidates.sort(key=lambda c: c["area_frac"], reverse=True)
    if MAX_PERSONS_PER_IMAGE is not None:
        for d in candidates[MAX_PERSONS_PER_IMAGE:]:
            d["status"] = "dropped (max_persons_per_image)"
        candidates = candidates[:MAX_PERSONS_PER_IMAGE]

    return candidates, diagnostics


def save_visualization(image, upper_body_masks, output_path):
    vis_img = image.convert("RGB")
    w, h = vis_img.size

    if not upper_body_masks:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        vis_img.save(output_path)
        return

    img_np = np.array(vis_img)
    overlay = img_np.copy()

    for mask in upper_body_masks:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        m8 = (mask * 255).astype(np.uint8)
        m8 = cv2.resize(m8, (w, h), interpolation=cv2.INTER_NEAREST)
        roi = m8 > 127
        overlay[roi] = (overlay[roi] * 0.6 + np.array([255, 140, 0]) * 0.4).astype(np.uint8)

    vis_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(vis_img)

    for i, mask in enumerate(upper_body_masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        m8 = cv2.resize((mask * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(m8 > 127)
        if len(xs) == 0:
            continue
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        draw.rectangle(box, outline="orange", width=2)
        draw.text((box[0] + 2, box[1] - 14), f"Upper Body {i}", fill="orange")

    draw.text((10, 10), "Orange = validated upper body (SAM-3 'upper body' prompt)", fill="white")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vis_img.save(output_path)


def main():
    print("Loading model...")
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID).to(device).eval()
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(SAM3_MODEL_ID)

    # Load images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(f for f in os.listdir(IMAGE_FOLDER)
                   if os.path.splitext(f)[1].lower() in valid_exts)[:100000]

    print(f"Processing {len(files)} images...\n")

    for idx, fname in enumerate(files, 1):
        img = Image.open(os.path.join(IMAGE_FOLDER, fname)).convert("RGB")
        inp = processor(images=img, text="person", return_tensors="pt").to(device)
        original_sizes = inp.get("original_sizes")
        original_sizes = original_sizes.tolist() if original_sizes is not None \
            else [[img.size[1], img.size[0]]]

        with torch.no_grad():
            ve = model.vision_encoder(inp["pixel_values"])
            tok_upper = tokenizer("head and upper body", return_tensors="pt").to(device)
            out_upper = model(vision_embeds=ve,
                              input_ids=tok_upper["input_ids"],
                              attention_mask=tok_upper["attention_mask"])
            tok_head = tokenizer("head", return_tensors="pt").to(device)
            out_head = model(vision_embeds=ve,
                             input_ids=tok_head["input_ids"],
                             attention_mask=tok_head["attention_mask"])

        upper_results = processor.post_process_instance_segmentation(
            out_upper, threshold=DETECTION_THRESHOLD, mask_threshold=MASK_THRESHOLD,
            target_sizes=original_sizes,
        )[0]
        head_results = processor.post_process_instance_segmentation(
            out_head, threshold=DETECTION_THRESHOLD, mask_threshold=MASK_THRESHOLD,
            target_sizes=original_sizes,
        )[0]

        upper_body_masks = [m.cpu() for m in upper_results["masks"]]
        upper_body_scores = upper_results["scores"].cpu().tolist()
        head_masks = [m.cpu() for m in head_results["masks"]]
        image_area = original_sizes[0][0] * original_sizes[0][1]
        candidates, diagnostics = filter_and_rank_people(
            upper_body_masks, head_masks, image_area, upper_body_scores=upper_body_scores)
        kept_upper_bodies = [c["mask"] for c in candidates]

        for d in diagnostics:
            score_str = f"{d['score']:.2f}" if d["score"] is not None else "n/a"
            if d["status"] == "kept":
                print(f"    -> person {d['index']}: KEPT score={score_str}, "
                      f"area_frac={d['area_frac']:.4f}")
            else:
                print(f"    -> person {d['index']}: {d['status']} ({d['reason']}), "
                      f"score={score_str}")

        out_path = os.path.join(OUTPUT_DIR, fname)
        save_visualization(img, kept_upper_bodies, out_path)

        print(f"  [{idx}/{len(files)}] {fname}: "
              f"{len(upper_body_masks)} upper-body detections, "
              f"{len(kept_upper_bodies)} kept")

    del model
    torch.cuda.empty_cache()
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()