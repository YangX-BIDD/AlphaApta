#!/usr/bin/env python3
"""Evaluate a final D2 checkpoint on an explicit held-out event array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from alphaapta.models import D2GNNTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--events",
        type=Path,
        required=True,
        help=(
            "NPZ with d1_states [N,L,256], aptamer_mask [N,L], "
            "cell_context [N,3257], and labels [N]."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arrays = np.load(args.events, mmap_mode="r")
    required = {"d1_states", "aptamer_mask", "cell_context", "labels"}
    missing = required - set(arrays.files)
    if missing:
        raise SystemExit(f"Missing event arrays: {sorted(missing)}")
    number = len(arrays["labels"])
    if not (
        len(arrays["d1_states"])
        == len(arrays["aptamer_mask"])
        == len(arrays["cell_context"])
        == number
    ):
        raise SystemExit("Event arrays have inconsistent row counts")

    device = torch.device(args.device)
    model = D2GNNTransformer().to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval()

    probabilities = []
    with torch.no_grad():
        for start in range(0, number, args.batch_size):
            stop = min(start + args.batch_size, number)
            logits, _, _ = model(
                torch.as_tensor(arrays["d1_states"][start:stop], device=device),
                torch.as_tensor(arrays["aptamer_mask"][start:stop], device=device),
                torch.as_tensor(arrays["cell_context"][start:stop], device=device),
            )
            probabilities.append(logits.softmax(-1)[:, 1].cpu().numpy())

    probability = np.concatenate(probabilities)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    predictions = (probability >= 0.5).astype(np.int64)
    result = {
        "number_of_events": int(number),
        "positive_events": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pr_auc": float(average_precision_score(labels, probability)),
        "f1_at_0.5": float(f1_score(labels, predictions)),
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
