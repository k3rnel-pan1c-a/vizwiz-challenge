"""
Visualization utilities for grounding predictions.

Draws predicted and ground-truth bounding boxes on images.
Saves qualitative examples organized by outcome type:
    - true_pos    : predicted box has IoU ≥ threshold with a GT box
    - false_pos   : predicted box has IoU < threshold with all GT boxes
    - false_neg   : GT box not covered by any predicted box
    - duplicate   : more than one predicted box matched to the same GT box
    - sm_conflict : single/multiple head disagrees with kept box count

All inputs use normalized cxcywh boxes; pixel conversion is done here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import Tensor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.boxes import cxcywh_to_xyxy, box_iou

# Color scheme
_GT_COLOR   = (0, 200, 0)     # green  — ground truth
_PRED_COLOR = (220, 50, 50)   # red    — prediction
_FP_COLOR   = (255, 165, 0)   # orange — false positive
_FN_COLOR   = (0, 100, 220)   # blue   — false negative
_DUP_COLOR  = (180, 0, 220)   # purple — duplicate prediction
_LINE_WIDTH = 3


# -----------------------------------------------------------------------
# Core drawing
# -----------------------------------------------------------------------

def draw_boxes(
    image: Image.Image,
    boxes_xyxy: list[list[float]] | Tensor,
    color: tuple[int, int, int],
    labels: list[str] | None = None,
    line_width: int = _LINE_WIDTH,
) -> Image.Image:
    """Draw bounding boxes on a PIL image (modifies a copy).

    Args:
        image:      PIL.Image.
        boxes_xyxy: list/tensor of [x1, y1, x2, y2] in pixel coordinates.
        color:      RGB tuple.
        labels:     optional text label per box.
        line_width: rectangle line thickness.

    Returns:
        New PIL.Image with boxes drawn.
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    if isinstance(boxes_xyxy, Tensor):
        boxes_xyxy = boxes_xyxy.cpu().tolist()

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=line_width)
        if labels and i < len(labels):
            draw.text((x1 + 2, y1 + 2), labels[i], fill=color + (255,))

    return Image.alpha_composite(img, overlay).convert("RGB")


def visualize_sample(
    image: Image.Image,
    pred_boxes_cxcywh: Tensor,
    gt_boxes_cxcywh: Tensor,
    pred_obj_scores: Tensor,
    obj_threshold: float = 0.5,
    single_multi_pred: str = "",
    single_multi_gt: str = "",
    title: str = "",
    iou_threshold: float = 0.5,
) -> Image.Image:
    """Draw both GT (green) and predictions (red/orange) on one image.

    Kept predictions (objectness > threshold) are shown in red if they
    match a GT box (IoU ≥ iou_threshold), orange otherwise.

    Args:
        image:               PIL.Image (RGB).
        pred_boxes_cxcywh:   Float[N, 4]  all predicted boxes.
        gt_boxes_cxcywh:     Float[K, 4]  GT boxes.
        pred_obj_scores:     Float[N]     sigmoid objectness probabilities.
        obj_threshold:       keep box if score > threshold.
        single_multi_pred:   string label for prediction ("single"/"multiple").
        single_multi_gt:     string label for GT.
        title:               text to prepend in top-left corner.
        iou_threshold:       IoU threshold to classify a kept box as TP.

    Returns:
        PIL.Image with annotations.
    """
    W, H = image.size

    # ---- convert boxes to pixel xyxy ----
    def to_pixel(boxes_norm: Tensor) -> Tensor:
        if boxes_norm.numel() == 0:
            return boxes_norm
        scale = torch.tensor([W, H, W, H], dtype=boxes_norm.dtype)
        return cxcywh_to_xyxy(boxes_norm) * scale

    gt_pixel   = to_pixel(gt_boxes_cxcywh)    # [K, 4]
    pred_pixel = to_pixel(pred_boxes_cxcywh)  # [N, 4]

    # ---- filter predictions by threshold ----
    keep = pred_obj_scores > obj_threshold
    kept_pred = pred_pixel[keep]
    kept_scores = pred_obj_scores[keep]

    # ---- classify kept predictions ----
    pred_colors: list[tuple[int, int, int]] = []
    if kept_pred.numel() > 0 and gt_pixel.numel() > 0:
        iou = box_iou(kept_pred, gt_pixel)   # [Nk, K]
        for i in range(len(kept_pred)):
            max_iou = iou[i].max().item() if gt_pixel.numel() > 0 else 0.0
            pred_colors.append(_PRED_COLOR if max_iou >= iou_threshold else _FP_COLOR)
    else:
        pred_colors = [_FP_COLOR] * len(kept_pred)

    img = image.copy()

    # Draw GT boxes
    if gt_pixel.numel() > 0:
        gt_labels = [f"GT{j}" for j in range(len(gt_pixel))]
        img = draw_boxes(img, gt_pixel, _GT_COLOR, labels=gt_labels)

    # Draw kept predicted boxes
    if len(kept_pred) > 0:
        for i, (box, score) in enumerate(zip(kept_pred.tolist(), kept_scores.tolist())):
            img = draw_boxes(img, [box], pred_colors[i],
                             labels=[f"P{i}:{score:.2f}"])

    # ---- overlay text header ----
    header = []
    if title:
        header.append(title)
    if single_multi_gt or single_multi_pred:
        header.append(f"GT={single_multi_gt}  Pred={single_multi_pred}")
    header.append(f"Kept={int(keep.sum())}  GT_boxes={len(gt_pixel)}")
    if header:
        img = _add_text_header(img, "\n".join(header))

    return img


def _add_text_header(image: Image.Image, text: str, padding: int = 4) -> Image.Image:
    """Paste a semi-transparent text bar at the top of the image."""
    draw = ImageDraw.Draw(image)
    lines = text.split("\n")
    line_h = 14
    bar_h = len(lines) * line_h + 2 * padding
    draw.rectangle([0, 0, image.width, bar_h], fill=(0, 0, 0, 160))
    for i, line in enumerate(lines):
        draw.text((padding, padding + i * line_h), line, fill=(255, 255, 255))
    return image


# -----------------------------------------------------------------------
# Qualitative example saver
# -----------------------------------------------------------------------

class QualitativeSaver:
    """Collect and save qualitative examples by outcome type."""

    def __init__(
        self,
        save_dir: str | Path,
        max_per_type: int = 20,
        iou_threshold: float = 0.5,
        obj_threshold: float = 0.5,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.max_per_type = max_per_type
        self.iou_threshold = iou_threshold
        self.obj_threshold = obj_threshold
        self._counts: dict[str, int] = {}

    def add(
        self,
        image: Image.Image,
        pred_boxes: Tensor,
        pred_scores: Tensor,
        gt_boxes: Tensor,
        sm_pred: int,
        sm_gt: int,
        image_id: str,
        question: str = "",
    ) -> None:
        """Classify the sample and save if slots remain."""
        W, H = image.size
        scale = torch.tensor([W, H, W, H], dtype=pred_boxes.dtype)

        keep = pred_scores > self.obj_threshold
        kept_pred = pred_boxes[keep]
        kept_scores = pred_scores[keep]
        n_kept = int(keep.sum())
        n_gt   = len(gt_boxes)

        sm_pred_str = "single" if sm_pred == 0 else "multiple"
        sm_gt_str   = "single" if sm_gt   == 0 else "multiple"

        # Determine outcome types for this sample
        outcome_types: list[str] = []

        if n_gt == 0:
            outcome_types.append("no_gt")
        elif n_kept == 0:
            outcome_types.append("false_neg")
        else:
            # Compute IoU between kept preds and GT
            kp_xyxy = cxcywh_to_xyxy(kept_pred) * scale
            gt_xyxy = cxcywh_to_xyxy(gt_boxes) * scale
            iou_mat = box_iou(kp_xyxy, gt_xyxy)   # [Nk, K]
            max_iou_per_pred = iou_mat.max(dim=1).values  # [Nk]
            max_iou_per_gt   = iou_mat.max(dim=0).values  # [K]

            has_tp = (max_iou_per_pred >= self.iou_threshold).any().item()
            has_fp = (max_iou_per_pred <  self.iou_threshold).any().item()
            has_fn = (max_iou_per_gt   <  self.iou_threshold).any().item()

            # Check duplicates: multiple preds matched to same GT
            matched_gt = iou_mat.argmax(dim=1)
            has_dup = False
            from collections import Counter
            c = Counter(matched_gt[max_iou_per_pred >= self.iou_threshold].tolist())
            if c and max(c.values()) > 1:
                has_dup = True

            if has_tp:
                outcome_types.append("true_pos")
            if has_fp:
                outcome_types.append("false_pos")
            if has_fn:
                outcome_types.append("false_neg")
            if has_dup:
                outcome_types.append("duplicate")

        # Single/multiple conflict
        sm_conflict = (
            (sm_pred == 0 and n_kept > 1)
            or (sm_pred == 1 and n_kept <= 1)
        )
        if sm_conflict:
            outcome_types.append("sm_conflict")

        for otype in outcome_types:
            cnt = self._counts.get(otype, 0)
            if cnt >= self.max_per_type:
                continue

            out_dir = self.save_dir / otype
            out_dir.mkdir(parents=True, exist_ok=True)

            title = f"ID={image_id} | Q: {question[:40]}"
            vis = visualize_sample(
                image=image,
                pred_boxes_cxcywh=pred_boxes,
                gt_boxes_cxcywh=gt_boxes,
                pred_obj_scores=pred_scores,
                obj_threshold=self.obj_threshold,
                single_multi_pred=sm_pred_str,
                single_multi_gt=sm_gt_str,
                title=title,
                iou_threshold=self.iou_threshold,
            )
            safe_id = image_id.replace("/", "_")
            vis.save(out_dir / f"{cnt:04d}_{safe_id}.png")
            self._counts[otype] = cnt + 1
