"""Cell-conditioned energy-based generator used by AlphaApta D3.

The default architecture and sampling parameters match the revised manuscript.
Lower scalar energy indicates greater compatibility between a relaxed nucleotide
state and its 256-dimensional cellular condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


BASES = ("", "A", "C", "G", "T")


@dataclass(frozen=True)
class D3EnergyConfig:
    sequence_length: int = 40
    vocabulary_size: int = 5
    condition_dimension: int = 256
    hidden_dimension: int = 256
    attention_heads: int = 4
    feedforward_dimension: int = 512
    transformer_layers: int = 3
    dropout: float = 0.1
    langevin_steps: int = 20
    langevin_gradient_step_size: float = 10.0
    langevin_noise_standard_deviation: float = 0.005
    energy_regularization: float = 0.1


class CellConditionedEnergyGenerator(nn.Module):
    """Assign sequence-condition energies and sample sequences by Langevin updates."""

    def __init__(self, config: D3EnergyConfig | None = None) -> None:
        super().__init__()
        self.config = config or D3EnergyConfig()
        cfg = self.config
        # Attribute names intentionally match the deposited checkpoint keys.
        self.in_proj = nn.Linear(cfg.vocabulary_size, cfg.hidden_dimension)
        self.pos = nn.Embedding(cfg.sequence_length, cfg.hidden_dimension)
        self.cond_proj = nn.Linear(cfg.condition_dimension, cfg.hidden_dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dimension,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dimension,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers)
        self.energy = nn.Sequential(
            nn.Linear(cfg.hidden_dimension, cfg.hidden_dimension),
            nn.GELU(),
            nn.Linear(cfg.hidden_dimension, 1),
        )

    def energy_value(
        self, relaxed_sequence: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """Return one scalar energy per sequence-condition pair."""
        cfg = self.config
        if relaxed_sequence.ndim != 3 or relaxed_sequence.shape[1:] != (
            cfg.sequence_length,
            cfg.vocabulary_size,
        ):
            raise ValueError(
                "relaxed_sequence must have shape "
                f"[batch, {cfg.sequence_length}, {cfg.vocabulary_size}]"
            )
        if condition.ndim != 2 or condition.shape[1] != cfg.condition_dimension:
            raise ValueError(
                f"condition must have shape [batch, {cfg.condition_dimension}]"
            )
        positions = torch.arange(cfg.sequence_length, device=relaxed_sequence.device)
        hidden = (
            self.in_proj(relaxed_sequence)
            + self.pos(positions)
            + self.cond_proj(condition).unsqueeze(1)
        )
        return self.energy(self.tf(hidden).mean(1)).squeeze(-1)

    def langevin_relaxation(
        self,
        condition: torch.Tensor,
        initial_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate relaxed particles by gradient descent in nucleotide-logit space."""
        cfg = self.config
        if initial_logits is None:
            logits = torch.zeros(
                condition.size(0),
                cfg.sequence_length,
                cfg.vocabulary_size,
                device=condition.device,
            )
        else:
            expected = (
                condition.size(0),
                cfg.sequence_length,
                cfg.vocabulary_size,
            )
            if tuple(initial_logits.shape) != expected:
                raise ValueError(f"initial_logits must have shape {expected}")
            logits = initial_logits.to(condition.device)
        logits = logits.detach().requires_grad_(True)
        for _ in range(cfg.langevin_steps):
            energy = self.energy_value(F.softmax(logits, dim=-1), condition).sum()
            (gradient,) = torch.autograd.grad(energy, logits)
            logits = (
                logits
                - cfg.langevin_gradient_step_size * gradient
                + cfg.langevin_noise_standard_deviation * torch.randn_like(logits)
            ).detach().requires_grad_(True)
        return F.softmax(logits.detach(), dim=-1)

    def loss(self, sequence_tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Contrast observed sequences with model-generated relaxed particles."""
        cfg = self.config
        positive = F.one_hot(
            sequence_tokens.long(), num_classes=cfg.vocabulary_size
        ).float()
        negative = self.langevin_relaxation(condition)
        positive_energy = self.energy_value(positive, condition)
        negative_energy = self.energy_value(negative, condition)
        regularizer = cfg.energy_regularization * (
            positive_energy.square().mean() + negative_energy.square().mean()
        )
        return positive_energy.mean() - negative_energy.mean() + regularizer

    @torch.no_grad()
    def generate(
        self, condition: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Sample complete 40-nt A/C/G/T sequences without causal decoding."""
        with torch.enable_grad():
            relaxed = self.langevin_relaxation(condition)
        logits = torch.log(relaxed + 1e-9) / max(float(temperature), 1e-6)
        logits[..., 0] = float("-inf")
        probabilities = logits.softmax(dim=-1).reshape(
            -1, self.config.vocabulary_size
        )
        return torch.multinomial(probabilities, 1).view(
            condition.size(0), self.config.sequence_length
        )


def decode_tokens(tokens: torch.Tensor) -> list[str]:
    """Decode token tensors using 0=PAD and 1/2/3/4=A/C/G/T."""
    return [
        "".join(BASES[index] for index in row if 0 < index < len(BASES))
        for row in tokens.tolist()
    ]
