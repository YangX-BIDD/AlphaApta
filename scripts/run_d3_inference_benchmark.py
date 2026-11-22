#!/usr/bin/env python3
"""Run the final D3 EBM on measured cellular contexts and audit its outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphaapta.runtime import AlphaAptaD3Runtime


def summarize(rows: list[dict]) -> dict:
    sequences = [row["sequence"] for row in rows]
    valid = [len(sequence) == 40 and set(sequence) <= set("ACGT") for sequence in sequences]
    return {
        "number_generated": len(sequences),
        "valid_fraction": sum(valid) / len(valid) if valid else 0.0,
        "unique_fraction": len(set(sequences)) / len(sequences) if sequences else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--contexts",
        type=Path,
        required=True,
        help="JSON object mapping cell-source labels to 3,257-value vectors.",
    )
    parser.add_argument("--per-context", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contexts = json.loads(args.contexts.read_text())
    if not isinstance(contexts, dict) or not contexts:
        raise SystemExit("--contexts must contain a non-empty JSON object")

    runtime = AlphaAptaD3Runtime(args.bundle, device=args.device)
    candidate_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for index, (cell_source, vector) in enumerate(contexts.items()):
        rows, metadata = runtime.generate(
            vector,
            n_candidates=args.per_context,
            cell_type=cell_source,
            temperature=args.temperature,
            seed=args.seed + index * 100_000,
        )
        for row in rows:
            row["intended_cell_source"] = cell_source
        candidate_rows.extend(rows)
        summaries[cell_source] = {**summarize(rows), "metadata": metadata}

    args.output.write_text(
        json.dumps(
            {
                "candidate_rows": candidate_rows,
                "per_context_summary": summaries,
                "overall_summary": summarize(candidate_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
