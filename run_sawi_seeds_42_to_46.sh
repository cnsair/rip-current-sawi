#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
out_dir="${1:-reviewer_interpolation/full_seeds42_46}"
exec python -u run_sawi_interpolation_control.py --seeds 42 43 44 45 46 --out "$out_dir"
