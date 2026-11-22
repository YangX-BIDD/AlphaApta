"""Final AlphaApta D2 GNN-Transformer aptamer-cell event classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class D2GNNTransformerConfig:
    d1_hidden_dimension: int = 256
    hidden_dimension: int = 256
    cell_context_dimension: int = 3257
    wnn_dimension: int = 50
    named_cell_features: int = 3207
    graph_neighbors: int = 8
    graph_layers: int = 2
    transformer_layers: int = 2
    cross_attention_layers: int = 2
    attention_heads: int = 8
    transformer_dropout: float = 0.1
    cross_attention_dropout: float = 0.0


def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class CellTokenizer(nn.Module):
    """Map each named RNA or AbSeq scalar to a learned feature token."""

    def __init__(self, hidden_dimension: int, number_of_features: int) -> None:
        super().__init__()
        self.emb = nn.Parameter(
            torch.randn(number_of_features, hidden_dimension) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(number_of_features, hidden_dimension))

    def forward(self, cell_context: torch.Tensor, offset: int) -> torch.Tensor:
        values = cell_context[:, offset:].unsqueeze(-1)
        return values * self.emb + self.bias


class XAttnFuse(nn.Module):
    """Two gated aptamer-to-cell cross-attention residual blocks."""

    def __init__(
        self,
        hidden_dimension: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_dimension, heads, dropout=dropout, batch_first=True
                )
                for _ in range(layers)
            ]
        )
        self.ln = nn.ModuleList(
            [nn.LayerNorm(hidden_dimension) for _ in range(layers)]
        )
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dimension, hidden_dimension * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dimension * 4, hidden_dimension),
                )
                for _ in range(layers)
            ]
        )
        self.ln2 = nn.ModuleList(
            [nn.LayerNorm(hidden_dimension) for _ in range(layers)]
        )
        self.alpha = nn.Parameter(torch.zeros(layers))

    def forward(
        self, aptamer_states: torch.Tensor, cell_tokens: torch.Tensor
    ) -> torch.Tensor:
        for index in range(len(self.attn)):
            attended, _ = self.attn[index](
                aptamer_states, cell_tokens, cell_tokens, need_weights=False
            )
            hidden = self.ln[index](
                aptamer_states + torch.tanh(self.alpha[index]) * attended
            )
            aptamer_states = self.ln2[index](hidden + self.ff[index](hidden))
        return aptamer_states


class D2GNNTransformer(nn.Module):
    """Predict binary aptamer-cell event logits from frozen D1 and cell inputs.

    Module names follow the deposited checkpoint exactly.
    """

    def __init__(self, config: D2GNNTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or D2GNNTransformerConfig()
        cfg = self.config
        hidden = cfg.hidden_dimension
        self.proj = nn.Sequential(
            nn.Linear(cfg.d1_hidden_dimension, hidden), nn.LayerNorm(hidden)
        )
        self.celltok = CellTokenizer(hidden, cfg.named_cell_features)
        self.fuse = XAttnFuse(
            hidden,
            cfg.attention_heads,
            cfg.cross_attention_layers,
            cfg.cross_attention_dropout,
        )
        self.cell_pool = nn.Sequential(
            nn.Linear(cfg.cell_context_dimension, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, hidden),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )
        self.gnn = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(cfg.graph_layers)]
        )
        self.gln = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(cfg.graph_layers)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=cfg.attention_heads,
            dim_feedforward=hidden * 4,
            dropout=cfg.transformer_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.transformer_layers
        )

    def encode_apt(
        self, aptamer_states: torch.Tensor, aptamer_mask: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        normalized = F.normalize(aptamer_states, dim=-1)
        similarity = normalized @ normalized.transpose(1, 2)
        similarity = similarity.masked_fill(
            ~aptamer_mask.unsqueeze(1).bool(), float("-inf")
        )
        neighbors = min(cfg.graph_neighbors, aptamer_states.size(1))
        indices = similarity.topk(neighbors, dim=-1).indices
        adjacency = torch.zeros_like(similarity).scatter_(-1, indices, 1.0)
        adjacency = adjacency * aptamer_mask.unsqueeze(1).to(adjacency.dtype)
        adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        hidden = aptamer_states
        for layer, norm in zip(self.gnn, self.gln):
            hidden = norm(hidden + F.relu(layer(adjacency @ hidden)))
        return self.enc(hidden, src_key_padding_mask=~aptamer_mask.bool())

    def forward(
        self,
        d1_states: torch.Tensor,
        aptamer_mask: torch.Tensor,
        cell_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        if cell_context.ndim != 2 or cell_context.size(1) != cfg.cell_context_dimension:
            raise ValueError(
                f"cell_context must have shape [batch, {cfg.cell_context_dimension}]"
            )
        aptamer_states = self.encode_apt(self.proj(d1_states.float()), aptamer_mask)
        cell_tokens = self.celltok(cell_context, cfg.wnn_dimension)
        fused_states = self.fuse(aptamer_states, cell_tokens)
        aptamer_representation = masked_mean(aptamer_states, aptamer_mask)
        fused_representation = masked_mean(fused_states, aptamer_mask)
        cell_representation = self.cell_pool(cell_context)
        logits = self.head(
            torch.cat([fused_representation, cell_representation], dim=-1)
        ) + 0.0 * self.logit_scale
        return logits, aptamer_representation, cell_representation


def d2_training_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    aptamer_representation: torch.Tensor,
    cell_representation: torch.Tensor,
    logit_scale: torch.Tensor,
    infonce_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Unweighted two-class CE plus symmetric positive-pair InfoNCE."""
    cross_entropy = F.cross_entropy(logits, labels.long())
    positive = labels.bool()
    if int(positive.sum()) > 1:
        scores = (
            logit_scale.exp().clamp(max=100)
            * F.normalize(aptamer_representation[positive], dim=-1)
            @ F.normalize(cell_representation[positive], dim=-1).T
        )
        targets = torch.arange(scores.size(0), device=scores.device)
        infonce = 0.5 * (
            F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets)
        )
    else:
        infonce = cross_entropy.new_zeros(())
    total = cross_entropy + float(infonce_weight) * infonce
    return total, {"cross_entropy": cross_entropy, "infonce": infonce}
