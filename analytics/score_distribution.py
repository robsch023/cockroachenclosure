#!/usr/bin/env python3
"""
Print the max cockroach-score for every validation image, grouped by true
label (positive/negative), so you can see where the two distributions sit
and choose a threshold that best separates them.

Usage:
    python score_distribution.py \
        --model model_float.tflite \
        --annotations dataset/validation/labels.json \
        --image-dir dataset/validation/images
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_labels(ann_path, image_dir):
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    annotated = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in annotated:
            annotated[iid] = ann["category_id"]

    labels = {}
    json_filenames = set()
    for image_id, file_name in id_to_filename.items():
        labels[file_name] = annotated.get(image_id, 0)
        json_filenames.add(file_name)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in sorted(Path(image_dir).iterdir()):
        if p.suffix.lower() in image_extensions and p.name not in json_filenames:
            labels[p.name] = 0

    return labels


def resolve_output_indices(interpreter):
    boxes_idx = scores_idx = None
    for d in interpreter.get_output_details():
        shape = list(d["shape"])
        if len(shape) == 3 and shape[-1] == 4:
            boxes_idx = d["index"]
        elif len(shape) == 3 and shape[-1] == 2:
            scores_idx = d["index"]
    if scores_idx is None:
        raise RuntimeError("Could not find scores tensor by shape.")
    return boxes_idx, scores_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-dir", required=True)
    args = parser.parse_args()

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        sys.exit("ai-edge-litert not found. Install with: pip install ai-edge-litert")

    labels = load_labels(args.annotations, args.image_dir)
    print(f"Total images: {len(labels)}  "
          f"positive: {sum(1 for v in labels.values() if v != 0)}  "
          f"negative: {sum(1 for v in labels.values() if v == 0)}")

    interpreter = Interpreter(model_path=args.model, num_threads=1)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    h, w = inp["shape"][1], inp["shape"][2]
    dtype = inp["dtype"]
    qp = inp.get("quantization_parameters", {})
    scales = qp.get("scales", [])
    zero_points = qp.get("zero_points", [])
    in_scale = float(scales[0]) if len(scales) > 0 else 1.0
    in_zp    = float(zero_points[0]) if len(zero_points) > 0 else 0.0

    boxes_idx, scores_idx = resolve_output_indices(interpreter)

    pos_scores = []
    neg_scores = []
    image_dir = Path(args.image_dir)

    for file_name, cat_id in labels.items():
        img_path = image_dir / file_name
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB").resize((w, h))
        arr = np.array(img, dtype=np.float32)
        if dtype == np.int8:
            arr = arr / 255.0
            arr = arr / in_scale + in_zp
            arr = np.clip(np.round(arr), -128, 127).astype(np.int8)
        elif dtype == np.uint8:
            arr = arr / 255.0
            arr = arr / in_scale + in_zp
            arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
        else:
            arr = arr / 255.0

        interpreter.set_tensor(inp["index"], arr[np.newaxis, ...])
        interpreter.invoke()

        sc_detail = [d for d in interpreter.get_output_details() if d["index"] == scores_idx][0]
        raw = interpreter.get_tensor(scores_idx)
        sqp = sc_detail.get("quantization_parameters", {})
        sscales = sqp.get("scales", [])
        szero = sqp.get("zero_points", [])
        s_scale = float(sscales[0]) if len(sscales) > 0 else 1.0
        s_zp    = float(szero[0]) if len(szero) > 0 else 0.0

        dequant = s_scale * (raw.astype(np.float32) - s_zp)
        max_score = float(np.max(dequant[0, :, 1]))

        if cat_id != 0:
            pos_scores.append(max_score)
        else:
            neg_scores.append(max_score)

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    print(f"\nPositive (cockroach) scores  — n={len(pos_scores)}")
    print(f"  min={pos_scores.min():.4f}  mean={pos_scores.mean():.4f}  "
          f"median={np.median(pos_scores):.4f}  max={pos_scores.max():.4f}")
    print(f"  percentiles: 5th={np.percentile(pos_scores,5):.4f}  "
          f"25th={np.percentile(pos_scores,25):.4f}")

    print(f"\nNegative (background) scores — n={len(neg_scores)}")
    print(f"  min={neg_scores.min():.4f}  mean={neg_scores.mean():.4f}  "
          f"median={np.median(neg_scores):.4f}  max={neg_scores.max():.4f}")
    print(f"  percentiles: 75th={np.percentile(neg_scores,75):.4f}  "
          f"95th={np.percentile(neg_scores,95):.4f}")

    # Suggest a threshold: try a range of candidates and report best separation
    print("\nThreshold sweep (precision/recall on positive class):")
    all_scores = np.concatenate([pos_scores, neg_scores])
    candidates = np.unique(np.round(all_scores, 3))
    best_f1, best_t = 0, 0
    print(f"{'Threshold':>10}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}  "
          f"{'Precision':>10}  {'Recall':>8}  {'F1':>6}")
    for t in candidates[::max(1, len(candidates)//20)]:  # sample ~20 thresholds
        tp = int((pos_scores >= t).sum())
        fn = int((pos_scores < t).sum())
        fp = int((neg_scores >= t).sum())
        tn = int((neg_scores < t).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
        print(f"{t:>10.4f}  {tp:>5}  {fp:>5}  {fn:>5}  {tn:>5}  "
              f"{precision:>10.4f}  {recall:>8.4f}  {f1:>6.4f}")

    print(f"\nBest F1 threshold from this sweep: {best_t:.4f} (F1={best_f1:.4f})")
    print("Note: choose based on YOUR priorities (e.g. minimizing missed cockroaches "
          "vs minimizing false door-openings), not blindly on F1.")


if __name__ == "__main__":
    main()
