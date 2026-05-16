"""
Loss functions for the VizWiz AnswerTherapy grounding model.

Total loss:
    L_total =
        lambda_box  * L1(matched pred boxes, matched GT boxes)
      + lambda_giou * (1 - GIoU(matched pred boxes, matched GT boxes))
      + lambda_obj  * BCE(all query objectness logits, target ∈ {0,1})
      + lambda_sm   * CE(pred_single_multi_logits, single_multi_labels)

All losses are averaged over the batch (number of samples with ≥1 GT box
for box/giou/obj, and all samples for single/multi CE).

Important:
- Matching indices come from HungarianMatcher (detached, no gradient).
- Losses are computed *after* indexing with those indices → differentiable.
- Objectness targets: matched queries → 1, unmatched queries → 0.
- pred_obj_logits are raw logits (BCE with logits).
- pred_single_multi_logits are raw logits (CE).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.boxes import cxcywh_to_xyxy, generalized_box_iou
from models.matcher import MatchResult


class LossDict(NamedTuple):
    total:   Tensor
    l1:      Tensor
    giou:    Tensor
    obj:     Tensor
    sm:      Tensor


class GroundingLoss(torch.nn.Module):
    """Compute the combined grounding loss.

    Args:
        lambda_box:  L1 box loss weight.
        lambda_giou: GIoU box loss weight.
        lambda_obj:  BCE objectness loss weight.
        lambda_sm:   CE single/multiple loss weight.
    """

    def __init__(
        self,
        lambda_box:  float = 5.0,
        lambda_giou: float = 2.0,
        lambda_obj:  float = 1.0,
        lambda_sm:   float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_box  = lambda_box
        self.lambda_giou = lambda_giou
        self.lambda_obj  = lambda_obj
        self.lambda_sm   = lambda_sm

    def forward(
        self,
        pred_boxes:       Tensor,
        pred_obj_logits:  Tensor,
        pred_sm_logits:   Tensor,
        gt_boxes_list:    list[Tensor],
        sm_labels:        Tensor,
        match_result:     MatchResult,
    ) -> LossDict:
        """Compute all losses.

        Args:
            pred_boxes      : Float[B, N, 4]  normalized cxcywh.
            pred_obj_logits : Float[B, N]     raw logits.
            pred_sm_logits  : Float[B, 2]     raw logits.
            gt_boxes_list   : list of Float[K_i, 4]  normalized cxcywh.
            sm_labels       : Long[B]  ground-truth single/multiple labels.
            match_result    : MatchResult from HungarianMatcher.

        Returns:
            LossDict with individual and total losses (all scalar tensors).
        """
        device = pred_boxes.device
        B, N, _ = pred_boxes.shape

        # ------------------------------------------------------------------ #
        # 1. Box losses (L1 + GIoU)  — only on matched pairs
        # ------------------------------------------------------------------ #
        l1_losses:   list[Tensor] = []
        giou_losses: list[Tensor] = []
        num_matched = 0

        for b in range(B):
            pred_idx = match_result.pred_indices[b]  # Long[M]
            gt_idx   = match_result.gt_indices[b]    # Long[M]
            M = pred_idx.numel()
            if M == 0:
                continue

            pb_matched = pred_boxes[b][pred_idx]              # [M, 4] cxcywh
            gt_boxes = gt_boxes_list[b].to(device)
            gt_matched = gt_boxes[gt_idx]                      # [M, 4] cxcywh

            # L1 loss on normalized cxcywh
            l1_losses.append(F.l1_loss(pb_matched, gt_matched, reduction="sum"))

            # GIoU loss: convert to xyxy first
            pb_xyxy = cxcywh_to_xyxy(pb_matched)      # [M, 4]
            gt_xyxy = cxcywh_to_xyxy(gt_matched)      # [M, 4]
            giou_mat = generalized_box_iou(pb_xyxy, gt_xyxy)  # [M, M]
            # Diagonal: matched pair GIoUs
            giou_diag = torch.diagonal(giou_mat)       # [M]
            giou_losses.append((1.0 - giou_diag).sum())

            num_matched += M

        if num_matched > 0:
            loss_l1   = torch.stack(l1_losses).sum()   / num_matched
            loss_giou = torch.stack(giou_losses).sum() / num_matched
        else:
            loss_l1   = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        # ------------------------------------------------------------------ #
        # 2. Objectness loss (BCE)  — all N queries, all B samples
        # ------------------------------------------------------------------ #
        obj_targets = torch.zeros(B, N, device=device, dtype=pred_boxes.dtype)
        for b in range(B):
            pred_idx = match_result.pred_indices[b]
            if pred_idx.numel() > 0:
                obj_targets[b, pred_idx] = 1.0

        loss_obj = F.binary_cross_entropy_with_logits(
            pred_obj_logits,
            obj_targets,
            reduction="mean",
        )

        # ------------------------------------------------------------------ #
        # 3. Single/multiple CE loss  — all samples
        # ------------------------------------------------------------------ #
        loss_sm = F.cross_entropy(pred_sm_logits, sm_labels.to(device))

        # ------------------------------------------------------------------ #
        # 4. Total
        # ------------------------------------------------------------------ #
        total = (
            self.lambda_box  * loss_l1
            + self.lambda_giou * loss_giou
            + self.lambda_obj  * loss_obj
            + self.lambda_sm   * loss_sm
        )

        return LossDict(
            total=total,
            l1=loss_l1.detach(),
            giou=loss_giou.detach(),
            obj=loss_obj.detach(),
            sm=loss_sm.detach(),
        )
