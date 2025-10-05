#!/usr/bin/env bash
set -euo pipefail
REPO=https://github.com/aiqm/torchani.git
DEST=${1:-external/torchani}
if [ -d "$DEST/.git" ]; then
  echo "[info] Existing torchani clone at $DEST" >&2
else
  git clone --depth 1 "$REPO" "$DEST"
fi
pip install -e "$DEST"
echo "[done] Installed torchani from $DEST"