#!/usr/bin/env python3
"""Run the deposited AlphaApta D1 sequence-frequency regressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from alphaapta.models import (
    D1SequenceFrequencyRegressor,
    destandardize_log10_frequency,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, metadata = D1SequenceFrequencyRegressor.from_checkpoint(
        args.checkpoint,
        model_name_or_path=args.model_directory,
        map_location="cpu",
    )
    tokenizer_source = str(args.model_directory or metadata.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True
    )
    encoded = tokenizer(
        [sequence.upper() for sequence in args.sequence],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    device = torch.device(args.device)
    model.to(device).eval()
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }
    with torch.no_grad():
        standardized = model(**encoded)
        log10_frequency = destandardize_log10_frequency(standardized, metadata)
    rows = [
        {
            "sequence": sequence,
            "standardized_prediction": float(prediction),
            "predicted_log10_frequency": float(log_frequency),
        }
        for sequence, prediction, log_frequency in zip(
            args.sequence,
            standardized.cpu(),
            log10_frequency.cpu(),
        )
    ]
    args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
