#!/bin/bash
# Install sprite_tools dependencies in Claude Code cloud sessions.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
pip install -q -r requirements.txt

# tesseract is optional (legend count check); never fail the session over it
if ! command -v tesseract >/dev/null 2>&1; then
  (apt-get update -qq && apt-get install -y -qq tesseract-ocr) >/dev/null 2>&1 \
    || echo "tesseract-ocr not installed; legend counts will not be checked" >&2
fi
