#!/usr/bin/env bash
# Install the exact paper implementation into a compatible VILA-HD checkout.
# This intentionally refuses to overwrite without an explicit acknowledgement.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VILA_ROOT="${VILA_ROOT:-}"
ACK="${1:-}"
EXPECTED_VILA_COMMIT="52a3735f7ac191113b5da44102b8e3a673b4dd32"

[[ -n "$VILA_ROOT" && -d "$VILA_ROOT/llava" ]] || {
    echo "Set VILA_ROOT to a compatible VILA-HD checkout." >&2
    exit 2
}
[[ "$ACK" == "I_HAVE_BACKED_UP_MY_VILA_CHECKOUT" ]] || {
    echo "Usage: VILA_ROOT=/path/to/VILA $0 I_HAVE_BACKED_UP_MY_VILA_CHECKOUT" >&2
    exit 2
}

if [[ -d "$VILA_ROOT/.git" ]]; then
    actual_commit="$(git -C "$VILA_ROOT" rev-parse HEAD)"
    if [[ "$actual_commit" != "$EXPECTED_VILA_COMMIT" && "${ALLOW_UNVERIFIED_VILA:-0}" != "1" ]]; then
        echo "Expected NVIDIA VILA commit: $EXPECTED_VILA_COMMIT" >&2
        echo "Checkout contains: $actual_commit" >&2
        echo "Refusing an unverified base. Set ALLOW_UNVERIFIED_VILA=1 only after auditing compatibility." >&2
        exit 3
    fi
else
    echo "VILA_ROOT must be a Git checkout so its base revision can be verified." >&2
    exit 3
fi

cp -R "$ROOT/src/vila_overlay/llava/." "$VILA_ROOT/llava/"
echo "Installed the paper overlay into: $VILA_ROOT"
