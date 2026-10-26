"""Interaction-conditioned nucleotide sequence decoder.

This module maps residue embeddings to nucleotide logits while using predicted
pair existence and optional pair classes as per-head attention biases.
"""

import torch
from torch import nn


class PairAwareTransformerSeqHead(nn.Module):
    """
    Sequence head that uses a small transformer over residues,
    with pair logits as attention bias.

    Inspired by Multiflow's sequence transformer blocks, but applied
    post-hoc to a frozen structure trunk.
    """

    def __init__(
        self,
        node_dim: int,
        num_classes: int,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_pair_bias: bool = True,
        pair_type_num_classes: int = 0,
        use_pair_type_bias: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.node_dim = node_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_pair_bias = use_pair_bias
        self.pair_type_num_classes = pair_type_num_classes
        self.use_pair_type_bias = use_pair_type_bias

        # Projections for attention.
        self.qkv_proj = nn.Linear(node_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, node_dim)

        # FFN.
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, node_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

        # Project scalar pair logits to per-head attention bias.
        if self.use_pair_bias:
            self.pair_bias_proj = nn.Linear(1, num_heads)
        else:
            self.pair_bias_proj = None
        if self.use_pair_type_bias:
            if pair_type_num_classes <= 0:
                raise ValueError("pair_type_num_classes must be positive when use_pair_type_bias=True")
            self.pair_type_bias_proj = nn.Linear(pair_type_num_classes, num_heads)
        else:
            self.pair_type_bias_proj = None

        # Final classification head.
        self.output_proj = nn.Linear(node_dim, num_classes)

    def forward(
        self,
        node_embed: torch.Tensor,
        pair_logits: torch.Tensor,
        res_mask: torch.Tensor,
        pair_type_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            node_embed: [B, L, node_dim]
            pair_logits: [B, L, L]
            res_mask: [B, L]
            pair_type_logits: [B, L, L, pair_type_num_classes], optional
        Returns:
            seq_logits: [B, L, num_classes]
        """
        B, L, D = node_embed.shape
        x = node_embed

        # Pair bias: [B, L, L, num_heads]
        pair_bias = None
        if self.use_pair_bias:
            pair_bias = self.pair_bias_proj(pair_logits.unsqueeze(-1))
            pair_bias = pair_bias.permute(0, 3, 1, 2)  # [B, H, L, L]
        if self.use_pair_type_bias and pair_type_logits is not None:
            pair_prob = torch.sigmoid(pair_logits).unsqueeze(-1)
            pair_type_prob = torch.softmax(pair_type_logits, dim=-1)
            typed_pair_feature = pair_prob * pair_type_prob
            pair_type_bias = self.pair_type_bias_proj(typed_pair_feature)
            pair_type_bias = pair_type_bias.permute(0, 3, 1, 2)  # [B, H, L, L]
            pair_bias = pair_type_bias if pair_bias is None else pair_bias + pair_type_bias

        # Attention mask from residue mask (key-only masking).
        # [B, L] -> [B, 1, 1, L]
        attn_mask = (~res_mask.bool()).unsqueeze(1).unsqueeze(2)

        for _ in range(self.num_layers):
            # Pre-norm.
            h = self.norm1(x)
            qkv = self.qkv_proj(h).reshape(
                B, L, 3, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)  # [3, B, H, L, d]
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
            if pair_bias is not None:
                attn_scores = attn_scores + pair_bias
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            attn_out = torch.matmul(attn_weights, v)  # [B, H, L, d]
            attn_out = attn_out.transpose(1, 2).reshape(B, L, self.hidden_dim)
            attn_out = self.out_proj(attn_out)

            x = x + attn_out
            x = x + self.ffn(self.norm2(x))

        # Apply mask before output.
        x = x * res_mask[..., None]
        logits = self.output_proj(x)
        return logits

