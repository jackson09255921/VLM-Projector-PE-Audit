#!/usr/bin/env bash
# Deterministic ten-benchmark evaluator for a compatible VILA-HD checkout.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VILA_ROOT="${VILA_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_BASE="${MODEL_BASE:-nvidia/VILA-HD-8B-PS3-1.5K-SigLIP2}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/eval}"
PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -n "$VILA_ROOT" ]] || { echo "VILA_ROOT is required" >&2; exit 2; }
[[ -n "$MODEL_PATH" ]] || { echo "MODEL_PATH is required" >&2; exit 2; }
[[ -n "${DATA_ROOT:-}" ]] || { echo "DATA_ROOT is required" >&2; exit 2; }

export VILA_DATASETS="$ROOT/configs/eval/datasets.example.yaml"
export DATA_ROOT
export PYTHONPATH="$VILA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"
cd "$VILA_ROOT"

run_one() {
    local name="$1" evaluator="$2" metric="$3" max_tokens="$4" split="${5:-}"
    local target="$OUTPUT_DIR/$name"
    local args=(
        "$PYTHON_BIN" "llava/eval/${evaluator}.py"
        --model-path "$MODEL_PATH" --model-base "$MODEL_BASE"
        --generation-config "{\"max_new_tokens\": ${max_tokens}, \"do_sample\": false}"
        --output-dir "$target"
    )
    [[ -z "$split" ]] || args+=(--split "$split")
    echo "[run] $name ($metric)"
    "${args[@]}"
}

run_one chartqa chartqa accuracy 32
run_one docvqa docvqa anls 64
run_one textvqa textvqa accuracy 16 val
run_one infovqa infovqa anls 64
run_one ocrbench ocrbench accuracy 32
run_one mathvista mathvista average.accuracy 128 testmini
run_one gqa gqa accuracy 16
run_one pope pope accuracy 8
run_one vstar vstar accuracy 8
run_one mmbench mmbench accuracy 8

echo "Evaluation complete: $OUTPUT_DIR"

