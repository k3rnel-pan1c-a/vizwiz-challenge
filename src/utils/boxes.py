"""
Bounding box utility functions.

All internal representations use normalized cxcywh in [0, 1].
Conversion to/from xyxy is done only for IoU/GIoU computation and visualization.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


# -----------------------------------------------------------------------
# Polygon → bounding box
# -----------------------------------------------------------------------

def polygon_to_xyxy(polygon: list[dict]) -> tuple[float, float, float, float]:
    """Convert a list of {x, y} points to an axis-aligned bounding box.

    Returns (x1, y1, x2, y2) in absolute pixel coordinates.
    """
    xs = [pt["x"] for pt in polygon]
    ys = [pt["y"] for pt in polygon]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def polygons_to_xyxy_list(polygons: list[list[dict]]) -> list[tuple[float, float, float, float]]:
    """Convert a list of polygons to a list of xyxy bounding boxes."""
    return [polygon_to_xyxy(p) for p in polygons]


def deduplicate_boxes(
    boxes_xyxy: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.85,
) -> list[tuple[float, float, float, float]]:
    """Remove near-duplicate boxes using greedy IoU-based deduplication.

    Boxes are added to the kept set in order; a box is dropped if it has
    IoU > iou_threshold with any already-kept box.
    """
    kept: list[tuple[float, float, float, float]] = []
    for box in boxes_xyxy:
        duplicate = False
        for ref in kept:
            if _iou_xyxy(box, ref) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def _iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two xyxy boxes (numpy-free, pure Python)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# -----------------------------------------------------------------------
# Coordinate format conversions
# -----------------------------------------------------------------------

def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert [x1, y1, x2, y2] → [cx, cy, w, h].

    Args:
        boxes: [..., 4] tensor in xyxy format.

    Returns:
        [..., 4] tensor in cxcywh format.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2].

    Args:
        boxes: [..., 4] tensor in cxcywh format.

    Returns:
        [..., 4] tensor in xyxy format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def normalize_xyxy(
    boxes_xyxy: Tensor | np.ndarray,
    width: float,
    height: float,
) -> Tensor:
    """Normalize xyxy boxes by image dimensions.

    Args:
        boxes_xyxy: [..., 4] in absolute pixel xyxy.
        width: image width in pixels.
        height: image height in pixels.

    Returns:
        [..., 4] normalized to [0, 1].
    """
    if isinstance(boxes_xyxy, np.ndarray):
        boxes_xyxy = torch.from_numpy(boxes_xyxy).float()
    scale = torch.tensor([width, height, width, height], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
    return boxes_xyxy / scale


def denormalize_cxcywh(
    boxes_cxcywh: Tensor,
    width: int,
    height: int,
) -> Tensor:
    """Convert normalized cxcywh to absolute pixel xyxy coordinates.

    Args:
        boxes_cxcywh: [..., 4] normalized cxcywh in [0, 1].
        width: image width in pixels.
        height: image height in pixels.

    Returns:
        [..., 4] absolute xyxy.
    """
    scale = torch.tensor([width, height, width, height], dtype=boxes_cxcywh.dtype, device=boxes_cxcywh.device)
    boxes_xyxy_norm = cxcywh_to_xyxy(boxes_cxcywh)
    return boxes_xyxy_norm * scale


def polygon_to_normalized_cxcywh(
    polygon: list[dict],
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Convert a polygon to normalized cxcywh.

    Returns (cx, cy, w, h) normalized to [0, 1].
    """
    x1, y1, x2, y2 = polygon_to_xyxy(polygon)
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return cx, cy, w, h


def clamp_boxes(boxes: Tensor) -> Tensor:
    """Clamp normalized cxcywh boxes to [0, 1] range, keeping center in image."""
    cx, cy, w, h = boxes.unbind(-1)
    cx = cx.clamp(0.0, 1.0)
    cy = cy.clamp(0.0, 1.0)
    w = w.clamp(0.0, 1.0)
    h = h.clamp(0.0, 1.0)
    return torch.stack([cx, cy, w, h], dim=-1)


# -----------------------------------------------------------------------
# IoU and GIoU (batch-friendly, work on xyxy format)
# -----------------------------------------------------------------------

def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes1: [N, 4] xyxy.
        boxes2: [M, 4] xyxy.

    Returns:
        [N, M] IoU matrix.
    """
    area1 = _box_area(boxes1)   # [N]
    area2 = _box_area(boxes2)   # [M]

    # [N, M, 2]
    inter_lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    inter_rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]    # [N, M]

    union = area1[:, None] + area2[None, :] - inter  # [N, M]
    iou = inter / union.clamp(min=1e-6)
    return iou


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise GIoU between two sets of xyxy boxes.

    Args:
        boxes1: [N, 4] xyxy.
        boxes2: [M, 4] xyxy.

    Returns:
        [N, M] GIoU matrix in [-1, 1].
    """
    iou = box_iou(boxes1, boxes2)    # [N, M]

    enclosing_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]   # [N, M]

    area1 = _box_area(boxes1)   # [N]
    area2 = _box_area(boxes2)   # [M]
    union = area1[:, None] + area2[None, :] - iou * (area1[:, None] + area2[None, :] - iou * enclosing_area)

    # Standard GIoU formula
    area1 = _box_area(boxes1)[:, None]   # [N, 1]
    area2 = _box_area(boxes2)[None, :]   # [1, M]
    inter_lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    inter_rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    union_area = area1 + area2 - inter_area

    enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]

    giou = inter_area / union_area.clamp(min=1e-6) - (enclosing_area - union_area) / enclosing_area.clamp(min=1e-6)
    return giou


def _box_area(boxes: Tensor) -> Tensor:
    """Area of xyxy boxes."""
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return w * h
