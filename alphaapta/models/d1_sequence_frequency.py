"""Sequence-frequency-supervised AlphaApta D1 model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class D1CheckpointMetadata:
    model_name: str
    target_mean: float
    target_standard_deviation: float
    epoch: int


class D1SequenceFrequencyRegressor(nn.Module):
    """HyenaDNA encoder with the scalar head used for D1 pretraining."""

    def __init__(self, encoder: nn.Module, hidden_dimension: int = 256) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        output = self.encoder(**inputs)
        states = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output[0]
        )
        return self.head(states.mean(dim=1)).squeeze(-1)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        model_name_or_path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple["D1SequenceFrequencyRegressor", D1CheckpointMetadata]:
        """Load the deposited D1 encoder, scalar head, and target scaling."""
        from transformers import AutoModel

        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        required = {"mid", "encoder", "head", "ym", "ys", "epoch"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"D1 checkpoint is missing fields: {sorted(missing)}")
        source = str(model_name_or_path or payload["mid"])
        try:
            encoder = AutoModel.from_pretrained(
                source, trust_remote_code=True, add_pooling_layer=False
            )
        except TypeError:
            encoder = AutoModel.from_pretrained(source, trust_remote_code=True)
        if hasattr(encoder, "pooler") and encoder.pooler is not None:
            encoder.pooler = None
        hidden = (
            getattr(encoder.config, "hidden_size", None)
            or getattr(encoder.config, "d_model", 256)
        )
        model = cls(encoder, hidden_dimension=int(hidden))
        model.encoder.load_state_dict(payload["encoder"], strict=True)
        model.head.load_state_dict(payload["head"], strict=True)
        metadata = D1CheckpointMetadata(
            model_name=str(payload["mid"]),
            target_mean=float(payload["ym"]),
            target_standard_deviation=float(payload["ys"]),
            epoch=int(payload["epoch"]),
        )
        return model, metadata


def destandardize_log10_frequency(
    standardized_prediction: torch.Tensor,
    metadata: D1CheckpointMetadata,
) -> torch.Tensor:
    """Convert a standardized D1 prediction back to log10 frequency."""
    return (
        standardized_prediction * metadata.target_standard_deviation
        + metadata.target_mean
    )
