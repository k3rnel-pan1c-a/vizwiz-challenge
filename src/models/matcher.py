"""
Hungarian matcher for DETR-style set prediction.

For each sample in the batch, builds a cost matrix [N, K] between
N predicted queries and K ground-truth boxes, then runs the linear
sum assignment (Hungarian algorithm) to find the optimal one-to-one
matching.

The matching is performed under torch.no_grad() so that the assignment
indices are detached from the computation graph. The losses computed
*after* matching (using the indices to index into predictions) are
still fully differentiable.

Cost matrix per sample:
    cost[i, j] = lambda_l1 * L1(pred_box_i, gt_box_j)
               + lambda_giou * (1 - GIoU(pred_box_i, gt_box_j))
               + lambda_obj * (-sigmoid(pred_obj_logit_i))

All boxes are in normalized cxcywh format internally; conversion to
xyxy is done only for GIoU computation.

When K == 0, no matching is performed and empty index tensors are returned.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.boxes import cxcywh_to_xyxy, generalized_box_iou


class MatchResult(NamedTuple):
    """Result of Hungarian matching for one batch.

    pred_indices: list of Long tensors, one per sample.
                  pred_indices[b][k] = which predicted query is matched to GT k.
    gt_indices:   list of Long tensors, one per sample.
                  gt_indices[b][k]   = which GT box is matched to predicted query k.
    Both tensors have the same length = number of matches for that sample.
    When K=0 both tensors are empty.
    """
    pred_indices: list[Tensor]
    gt_indices:   list[Tensor]


class HungarianMatcher(torch.nn.Module):
    """Compute optimal Hungarian assignment between N queries and K GT boxes.

    Args:
        lambda_l1:   weight for L1 bounding box cost.
        lambda_giou: weight for GIoU cost.
        lambda_obj:  weight for objectness cost.
    """

    def __init__(
        self,
        lambda_l1: float = 5.0,
        lambda_giou: float = 2.0,
        lambda_obj: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_obj = lambda_obj

    @torch.no_grad()
    def forward(
        self,
        pred_boxes: Tensor,
        pred_obj_logits: Tensor,
        gt_boxes_list: list[Tensor],
    ) -> MatchResult:
        """Compute Hungarian matching for a batch.

        Args:
            pred_boxes       : Float[B, N, 4]  normalized cxcywh.
            pred_obj_logits  : Float[B, N]     raw objectness logits.
            gt_boxes_list    : list of Float[K_i, 4] tensors (one per sample).
                               K_i may be 0.

        Returns:
            MatchResult with pred_indices and gt_indices lists.
        """
        B = pred_boxes.size(0)
        all_pred_idx: list[Tensor] = []
        all_gt_idx:   list[Tensor] = []

        for b in range(B):
            pb   = pred_boxes[b]       # [N, 4] cxcywh normalized
            pol  = pred_obj_logits[b]  # [N]
            gt_b = gt_boxes_list[b].to(pb.device)   # [K, 4]

            K = gt_b.size(0)
            if K == 0:
                empty = torch.zeros(0, dtype=torch.long, device=pb.device)
                all_pred_idx.append(empty)
                all_gt_idx.append(empty)
                continue

            N = pb.size(0)

            # ---- L1 cost  [N, K] ----
            cost_l1 = torch.cdist(pb, gt_b, p=1)   # [N, K]

            # ---- GIoU cost  [N, K] ----
            pb_xyxy = cxcywh_to_xyxy(pb)     # [N, 4]
            gt_xyxy = cxcywh_to_xyxy(gt_b)   # [K, 4]
            giou = generalized_box_iou(pb_xyxy, gt_xyxy)   # [N, K]
            cost_giou = 1.0 - giou

            # ---- Objectness cost  [N, K] (broadcast same value per row) ----
            obj_prob = torch.sigmoid(pol)                        # [N]
            cost_obj = -obj_prob.unsqueeze(1).expand(N, K)       # [N, K]

            # ---- Combined cost ----
            cost = (
                self.lambda_l1   * cost_l1
                + self.lambda_giou * cost_giou
                + self.lambda_obj  * cost_obj
            )   # [N, K]

            # ---- Solve assignment ----
            cost_np = cost.cpu().float().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            pred_idx = torch.as_tensor(row_ind, dtype=torch.long, device=pb.device)
            gt_idx   = torch.as_tensor(col_ind, dtype=torch.long, device=pb.device)

            all_pred_idx.append(pred_idx)
            all_gt_idx.append(gt_idx)

        return MatchResult(pred_indices=all_pred_idx, gt_indices=all_gt_idx)
