#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQ_FILE="${1:-requirements.lite.txt}"

if [[ ! -f "$SCRIPT_DIR/$REQ_FILE" ]]; then
  echo "Error: $SCRIPT_DIR/$REQ_FILE not found"
  exit 1
fi

echo "==> Removing old venv (if any)..."
rm -rf "$VENV_DIR"

echo "==> Creating fresh venv..."
python3 -m venv "$VENV_DIR"

echo "==> Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

echo "==> Installing from $REQ_FILE..."
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/$REQ_FILE"

echo ""
echo "Done! Activate with:"
echo "  source $VENV_DIR/bin/activate"
