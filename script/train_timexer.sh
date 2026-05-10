#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

conda run -n TimeXer python -m fsdownload.adjust.train_timexer "$@"
