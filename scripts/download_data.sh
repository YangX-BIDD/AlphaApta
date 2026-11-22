#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${repository_root}/scripts/download_zenodo_record.py" \
  --record 18242287 \
  --output "${repository_root}/downloads" \
  "$@"
