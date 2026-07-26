#!/usr/bin/env bash
# Install the V5 memory plugin (cross-platform wrapper around install.py).
# Usage: ./install.sh [--download-models] [--hermes-agent /path/to/hermes-agent] ...
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python
fi

exec "$PY" "$HERE/install.py" "$@"
