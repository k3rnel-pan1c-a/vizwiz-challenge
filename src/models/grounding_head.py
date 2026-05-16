"""
DETR-style grounding head.

Architecture:
    Qwen2.5-VL hidden states  [B, L, d_vlm]
        ↓  input_proj  (Linear + LayerNorm)
    Memory                    [B, L, d_dec]
        ↓  TransformerDecoder  (N learnable queries attend to Memory)
    Query outputs             [B, N, d_dec]
        ├─ box_head    → pred_boxes           [B, N, 4]   normalized cxcywh
        ├─ obj_head    → pred_obj_logits      [B, N]      raw logits
        └─ sm_head     → pred_single_multi_logits [B, 2]  raw logits

Outputs are raw logits (no sigmoid / softmax applied here).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# -----------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------

class MLP(nn.Module):
    """Simple multi-layer perceptron with ReLU activations."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.num_layers = num_layers

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncoding1D(nn.Module):
    """Sinusoidal 1-D positional encoding added to memory (VLM sequence).

    Optional — improves position awareness of decoder cross-attention.
    """

    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)   # [max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L, d_model]"""
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


# -----------------------------------------------------------------------
# Main module
# -----------------------------------------------------------------------

class GroundingHead(nn.Module):
    """DETR-style grounding decoder that takes VLM hidden states as memory.

    Args:
        d_vlm: dimensionality of the VLM backbone's hidden states.
        d_dec: hidden dimensionality of the DETR decoder.
        num_queries: number of learnable object queries (N).
        num_decoder_layers: number of Transformer decoder layers.
        nheads: number of attention heads.
        dropout: dropout rate in the Transformer.
        use_single_multi_head: whether to include the auxiliary
            single/multiple classification head.
        use_pos_enc: add sinusoidal positional encoding to memory.
    """

    def __init__(
        self,
        d_vlm: int = 3584,
        d_dec: int = 256,
        num_queries: int = 8,
        num_decoder_layers: int = 6,
        nheads: int = 8,
        dropout: float = 0.1,
        use_single_multi_head: bool = True,
        use_pos_enc: bool = True,
    ) -> None:
        super().__init__()

        self.d_dec = d_dec
        self.num_queries = num_queries
        self.use_single_multi_head = use_single_multi_head

        # ---- Memory projection ----
        self.input_proj = nn.Sequential(
            nn.Linear(d_vlm, d_dec),
            nn.LayerNorm(d_dec),
        )

        # ---- Optional positional encoding for memory ----
        self.pos_enc: Optional[nn.Module] = PositionalEncoding1D(d_dec) if use_pos_enc else None

        # ---- Learnable object queries ----
        self.query_embed = nn.Embedding(num_queries, d_dec)

        # ---- Transformer decoder ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_dec,
            nhead=nheads,
            dim_feedforward=d_dec * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,   # [B, seq, d_model] convention
            norm_first=True,    # Pre-LN for training stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_dec),
        )

        # ---- Output heads ----
        # Box head: MLP → 4, sigmoid applied in forward (cxcywh in [0,1])
        self.box_head = MLP(d_dec, d_dec, 4, num_layers=3)

        # Objectness head: Linear → scalar raw logit per query
        self.obj_head = nn.Linear(d_dec, 1)

        # Single/multiple head: pooled query features → 2 logits
        if use_single_multi_head:
            self.sm_head = MLP(d_dec, d_dec // 2, 2, num_layers=2)
        else:
            self.sm_head = None

        self._init_weights()

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            hidden_states    : Float[B, L, d_vlm]  — VLM last hidden states.
            key_padding_mask : Bool[B, L] or None  — True = padding position.

        Returns dict with:
            pred_boxes              : Float[B, N, 4]  normalized cxcywh in [0,1]
            pred_obj_logits         : Float[B, N]     raw objectness logits
            pred_single_multi_logits: Float[B, 2]     raw logits (0=single, 1=multi)
                                      (zeros if use_single_multi_head=False)
        """
        B = hidden_states.size(0)

        # ---- Project memory ----
        memory = self.input_proj(hidden_states)   # [B, L, d_dec]
        if self.pos_enc is not None:
            memory = self.pos_enc(memory)

        # ---- Expand object queries for the batch ----
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)   # [B, N, d_dec]

        # ---- Transformer decoder: queries attend to memory ----
        # PyTorch TransformerDecoder convention (batch_first=True):
        #   tgt  = [B, N, d_dec]
        #   memory = [B, L, d_dec]
        #   memory_key_padding_mask = [B, L] bool (True = ignore)
        decoder_out = self.transformer_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=key_padding_mask,
        )   # [B, N, d_dec]

        # ---- Prediction heads ----
        # Boxes: sigmoid to ensure output in (0, 1)
        pred_boxes = self.box_head(decoder_out).sigmoid()           # [B, N, 4]

        # Objectness: raw logits
        pred_obj_logits = self.obj_head(decoder_out).squeeze(-1)    # [B, N]

        # Single/multiple: pool over queries, then classify
        if self.sm_head is not None:
            pooled = decoder_out.mean(dim=1)                        # [B, d_dec]
            pred_sm_logits = self.sm_head(pooled)                   # [B, 2]
        else:
            pred_sm_logits = torch.zeros(B, 2, device=hidden_states.device, dtype=hidden_states.dtype)

        return {
            "pred_boxes": pred_boxes,
            "pred_obj_logits": pred_obj_logits,
            "pred_single_multi_logits": pred_sm_logits,
        }

    # ------------------------------------------------------------------ #
    # Weight initialization
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Bias init for box head: center predictions to image center
        # (sigmoid(0) = 0.5 → center; bias=-log(1/0.5 - 1) = 0)
        nn.init.constant_(self.box_head.layers[-1].bias, 0.0)

        # Object head: start with low confidence (negative bias)
        nn.init.constant_(self.obj_head.bias, -2.0)


# -----------------------------------------------------------------------
# Full model convenience wrapper
# -----------------------------------------------------------------------

class VizWizGroundingModel(nn.Module):
    """Combined backbone + grounding head.

    Instantiate backbone separately and pass it in so that its weights
    live on the right device map.  The grounding head is always on a
    single device (typically cuda:0 or the last GPU in a device_map).
    """

    def __init__(self, backbone: nn.Module, head: GroundingHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        images: list,
        questions: list[str],
        device: torch.device | None = None,
    ) -> dict[str, Tensor]:
        """End-to-end forward pass.

        Returns the same dict as GroundingHead.forward().
        """
        hidden_states, key_pad_mask = self.backbone(images, questions, device=device)
        # Move to head's device if they differ (multi-GPU scenarios)
        head_device = next(self.head.parameters()).device
        hidden_states = hidden_states.to(head_device)
        key_pad_mask = key_pad_mask.to(head_device)
        return self.head(hidden_states, key_padding_mask=key_pad_mask)
