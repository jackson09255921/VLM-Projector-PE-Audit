#!/usr/bin/env bash
# Install the exact paper implementation into a compatible VILA-HD checkout.
# This intentionally refuses to overwrite without an explicit acknowledgement.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VILA_ROOT="${VILA_ROOT:-}"
ACK="${1:-}"

[[ -n "$VILA_ROOT" && -d "$VILA_ROOT/llava" ]] || {
    echo "Set VILA_ROOT to a compatible VILA-HD checkout." >&2
    exit 2
}
[[ "$ACK" == "I_HAVE_BACKED_UP_MY_VILA_CHECKOUT" ]] || {
    echo "Usage: VILA_ROOT=/path/to/VILA $0 I_HAVE_BACKED_UP_MY_VILA_CHECKOUT" >&2
    exit 2
}

cp -R "$ROOT/src/vila_overlay/llava/." "$VILA_ROOT/llava/"
echo "Installed the paper overlay into: $VILA_ROOT"

