"""
Inference script for VizWiz AnswerTherapy grounding model.

Given an image (or a directory of images) and a question, runs the
model and outputs:
    - Annotated image with kept bounding boxes drawn on it
    - JSON with structured predictions

No text generation is used for boxes. All boxes come from
objectness thresholding of the DETR decoder output.

Usage — single image:
    python src/infer.py \
        --config configs/base.yaml \
        --checkpoint outputs/checkpoints/latest.pt \
        --image path/to/image.jpg \
        --question "What is this?" \
        [--obj_threshold 0.5] \
        [--out_dir outputs/inference]

Usage — batch from JSON:
    python src/infer.py \
        --config configs/base.yaml \
        --checkpoint outputs/checkpoints/latest.pt \
        --json_file data/my_samples.json \
        --img_dir data/images \
        [--obj_threshold 0.5]

JSON file format (list of objects):
    [{"image_id": "foo.jpg", "question": "What is this?"}, ...]

Output JSON per sample:
    {
        "image_id": "...",
        "question": "...",
        "single_multi_prediction": "single" | "multiple",
        "single_multi_probabilities": {"single": float, "multiple": float},
        "kept_boxes_xyxy": [[x1,y1,x2,y2], ...],
        "kept_box_scores": [float, ...],
        "num_kept_boxes": int
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.qwen_backbone import QwenBackbone
from models.grounding_head import GroundingHead
from utils.boxes import cxcywh_to_xyxy
from utils.visualization import draw_boxes
from train import load_config, build_model


_SM_LABELS = {0: "single", 1: "multiple"}


# -----------------------------------------------------------------------
# Core inference function
# -----------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    backbone: QwenBackbone,
    head: GroundingHead,
    images: list[Image.Image],
    questions: list[str],
    image_ids: list[str],
    device: torch.device,
    obj_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Run model on a batch and return structured predictions.

    Args:
        images:         list of PIL.Image (batch).
        questions:      list of question strings.
        image_ids:      list of image identifiers (for output metadata).
        obj_threshold:  sigmoid objectness score threshold for keeping boxes.

    Returns:
        List of per-sample prediction dicts.
    """
    hidden, key_pad = backbone(images, questions, device=device)
    hidden  = hidden.to(device)
    key_pad = key_pad.to(device)

    out = head(hidden, key_padding_mask=key_pad)
    pred_boxes_batch = out["pred_boxes"]      # [B, N, 4] cxcywh normalized
    pred_obj_batch   = out["pred_obj_logits"] # [B, N]
    pred_sm_batch    = out["pred_single_multi_logits"]  # [B, 2]

    results = []
    for b, (img, q, iid) in enumerate(zip(images, questions, image_ids)):
        W, H = img.size

        # ---- objectness filtering ----
        scores = torch.sigmoid(pred_obj_batch[b])   # [N]
        keep   = scores > obj_threshold
        kept_boxes  = pred_boxes_batch[b][keep]     # [Nk, 4] cxcywh
        kept_scores = scores[keep]                  # [Nk]

        # ---- convert to absolute xyxy ----
        if kept_boxes.numel() > 0:
            scale = torch.tensor([W, H, W, H], dtype=kept_boxes.dtype, device=kept_boxes.device)
            kept_xyxy = cxcywh_to_xyxy(kept_boxes) * scale
            kept_xyxy_list = kept_xyxy.cpu().tolist()
            kept_scores_list = kept_scores.cpu().tolist()
        else:
            kept_xyxy_list   = []
            kept_scores_list = []

        # ---- single/multiple ----
        sm_probs = F.softmax(pred_sm_batch[b].float(), dim=0)
        sm_pred  = int(sm_probs.argmax())
        sm_label = _SM_LABELS[sm_pred]

        results.append({
            "image_id":   iid,
            "question":   q,
            "single_multi_prediction": sm_label,
            "single_multi_probabilities": {
                "single":   round(float(sm_probs[0]), 4),
                "multiple": round(float(sm_probs[1]), 4),
            },
            "kept_boxes_xyxy": [[round(v, 2) for v in box] for box in kept_xyxy_list],
            "kept_box_scores": [round(float(s), 4) for s in kept_scores_list],
            "num_kept_boxes": len(kept_xyxy_list),
            # Internal fields for visualization
            "_image":       img,
            "_kept_boxes":  kept_xyxy_list,
            "_kept_scores": kept_scores_list,
        })

    return results


# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------

def save_visualization(
    result: dict,
    out_dir: Path,
) -> None:
    """Draw kept boxes on image and save."""
    img = result["_image"]
    boxes = result["_kept_boxes"]
    scores = result["_kept_scores"]

    labels = [f"{s:.2f}" for s in scores] if scores else []
    if boxes:
        from utils.visualization import draw_boxes
        img = draw_boxes(img, boxes, color=(220, 50, 50), labels=labels)

    # Header text
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    header = (
        f"SM={result['single_multi_prediction']} "
        f"(p={result['single_multi_probabilities'][result['single_multi_prediction']]:.2f})  "
        f"kept={result['num_kept_boxes']}"
    )
    draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    draw.text((4, 2), header, fill=(255, 255, 255))

    safe_id = result["image_id"].replace("/", "_")
    out_path = out_dir / f"{safe_id}_pred.png"
    img.save(out_path)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/base.yaml")
    parser.add_argument("--checkpoint",    required=True)
    # Single-image mode
    parser.add_argument("--image",         default=None)
    parser.add_argument("--question",      default=None)
    # Batch mode
    parser.add_argument("--json_file",     default=None)
    parser.add_argument("--img_dir",       default=None)
    # Common options
    parser.add_argument("--obj_threshold", type=float, default=None)
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--out_dir",       default=None)
    parser.add_argument("--save_viz",      action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    obj_thr = args.obj_threshold or cfg["eval"].get("obj_threshold", 0.5)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir or cfg["output"].get("viz_dir", "outputs/inference"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build model ----
    backbone, head = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')})")

    # ---- Collect samples ----
    samples: list[tuple[str, str]] = []   # (image_path, question)

    if args.image and args.question:
        samples = [(args.image, args.question)]
    elif args.json_file:
        with open(args.json_file) as f:
            raw = json.load(f)
        img_dir = Path(args.img_dir) if args.img_dir else Path(".")
        for item in raw:
            image_id = str(item["image_id"])
            img_path = str(img_dir / image_id) if not image_id.endswith((".jpg", ".png")) else str(img_dir / image_id)
            samples.append((img_path, item.get("question", "")))
    else:
        parser.error("Provide either --image + --question or --json_file")

    # ---- Run inference in batches ----
    all_results: list[dict] = []
    batch_size = args.batch_size

    for start in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        batch_samples = samples[start : start + batch_size]
        batch_images  = []
        batch_qs      = []
        batch_ids     = []

        for img_path, q in batch_samples:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"WARNING: could not load {img_path}: {e}")
                continue
            batch_images.append(img)
            batch_qs.append(q)
            batch_ids.append(Path(img_path).name)

        if not batch_images:
            continue

        results = run_inference(
            backbone=backbone, head=head,
            images=batch_images, questions=batch_qs, image_ids=batch_ids,
            device=device, obj_threshold=obj_thr,
        )
        all_results.extend(results)

        if args.save_viz:
            for r in results:
                save_visualization(r, out_dir)

    # ---- Strip internal PIL fields before saving JSON ----
    clean_results = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in all_results
    ]

    json_path = out_dir / "predictions.json"
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nSaved {len(clean_results)} predictions to {json_path}")

    # ---- Print first few results ----
    for r in clean_results[:3]:
        print(f"\n  Image: {r['image_id']}")
        print(f"  Question: {r['question']}")
        print(f"  SM prediction: {r['single_multi_prediction']}  "
              f"(probs: {r['single_multi_probabilities']})")
        print(f"  Kept boxes ({r['num_kept_boxes']}): {r['kept_boxes_xyxy']}")
        print(f"  Scores: {r['kept_box_scores']}")


if __name__ == "__main__":
    main()
