#!/usr/bin/env python3
"""Generate AlphaApta D3 candidates from a 3,257-value JSON cell vector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphaapta.runtime import AlphaAptaD3Runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cell-context", type=Path, required=True)
    parser.add_argument("--cell-type")
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    context = json.loads(args.cell_context.read_text())
    if isinstance(context, dict):
        context = context["combined_expression"]
    runtime = AlphaAptaD3Runtime(args.bundle, device=args.device)
    rows, metadata = runtime.generate(
        context,
        n_candidates=args.number,
        cell_type=args.cell_type,
        temperature=args.temperature,
        seed=args.seed,
    )
    args.output.write_text(json.dumps({"metadata": metadata, "candidates": rows}, indent=2))


if __name__ == "__main__":
    main()
