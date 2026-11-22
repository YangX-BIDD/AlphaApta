#!/usr/bin/env python3
"""Download files exposed by a public Zenodo record without hard-coded names."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Download only a named file; may be supplied more than once.",
    )
    args = parser.parse_args()

    record_id = str(args.record).strip()
    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        with urllib.request.urlopen(api_url) as response:
            record = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        # Resolve Concept DOI / Concept ID redirect (e.g. 18242287 -> latest version record ID)
        with urllib.request.urlopen(f"https://zenodo.org/records/{record_id}") as redir:
            resolved_id = redir.geturl().rstrip("/").split("/")[-1]
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{resolved_id}") as response:
            record = json.load(response)

    files = record.get("files", [])
    requested = set(args.file)
    if requested:
        files = [item for item in files if item.get("key") in requested]
        missing = requested - {item.get("key") for item in files}
        if missing:
            raise SystemExit(f"Files not present in record: {sorted(missing)}")
    if not files:
        raise SystemExit("The Zenodo record currently exposes no downloadable files.")

    args.output.mkdir(parents=True, exist_ok=True)
    for item in files:
        name = item["key"]
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = item["links"].get("content") or item["links"]["self"]
        print(f"Downloading {name} -> {target}")
        urllib.request.urlretrieve(url, target)


if __name__ == "__main__":
    main()
