#!/usr/bin/env bash
# Reproduce one matched-312 training run in a compatible VILA-HD checkout.
# The release repository is an overlay; run scripts/install_overlay.sh first.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VILA_ROOT="${VILA_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-}"
VARIANT="${VARIANT:-F}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/${VARIANT}-seed${SEED}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-nvidia/VILA-HD-8B-PS3-1.5K-SigLIP2}"
VISION_TOWER="${VISION_TOWER:-nvidia/PS3_Lang-1.5K-SigLIP2}"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: VARIANT=F SEED=42 VILA_ROOT=/path/to/VILA DATA_ROOT=/path/to/data $0 [--dry-run]" >&2
    exit 2
fi

[[ -n "$VILA_ROOT" ]] || { echo "VILA_ROOT is required" >&2; exit 2; }
[[ "$DRY_RUN" == true || -n "$DATA_ROOT" ]] || { echo "DATA_ROOT is required" >&2; exit 2; }

if [[ "$DRY_RUN" == false ]]; then
    [[ -f "$VILA_ROOT/llava/train/train_mem.py" ]] || {
        echo "VILA_ROOT does not contain llava/train/train_mem.py" >&2
        exit 3
    }
    [[ -d "$DATA_ROOT" ]] || { echo "DATA_ROOT is not a directory" >&2; exit 3; }
    [[ ! -e "$OUTPUT_DIR" ]] || {
        echo "Refusing to overwrite an existing output: $OUTPUT_DIR" >&2
        exit 3
    }
fi

case "$VARIANT" in
    N) PE_ARGS=(--pos_embed_type none) ;;
    A) PE_ARGS=(--pos_embed_type learned --reinit_pos_embed True) ;;
    A0) PE_ARGS=(--pos_embed_type learned --zero_init_pos_embed True) ;;
    C) PE_ARGS=(--pos_embed_type fourier --pos_embed_num_freqs 32) ;;
    E) PE_ARGS=(--pos_embed_type log_retina --pos_embed_alpha 1.0 --pos_embed_num_freqs 32 --pos_embed_dynamic_center True) ;;
    F) PE_ARGS=(--pos_embed_type polar --pos_embed_num_freqs 32) ;;
    *) echo "VARIANT must be one of N, A, A0, C, E, F" >&2; exit 2 ;;
esac

MIXTURE="chartqa_train@12k+docvqa_train@12k+shareGPT4V@25k+svit_complex_reasoning@12k+svit_conversation@12k+tulu@8k"
COMMAND=(
    "$PYTHON_BIN" llava/train/train_mem.py
    --model_name_or_path "$BASE_MODEL"
    --vision_tower "$VISION_TOWER"
    --data_mixture "$MIXTURE"
    --training_stage stage3 --chat_template auto
    --output_dir "$OUTPUT_DIR/model" --run_name "projector-pe-${VARIANT}-seed${SEED}"
    --bf16 True --bits 4 --optim paged_adamw_8bit
    --mm_projector mlp_downsample --mm_vision_select_feature cls_patch --mm_vision_select_layer -2
    --image_aspect_ratio resize --mm_use_im_start_end False --mm_use_im_patch_token False
    --ps3 True --look_close_mode after_prompt --num_look_close 3
    --high_res_pos_embed True "${PE_ARGS[@]}"
    --ps3_dynamic_aspect_ratio False --high_res_size 1512
    --max_steps 312 --seed "$SEED"
    --per_device_train_batch_size 1 --gradient_accumulation_steps 64
    --learning_rate 2e-5 --mm_projector_lr 2e-4 --weight_decay 0.0
    --warmup_ratio 0.03 --lr_scheduler_type cosine --max_grad_norm 1.0
    --model_max_length 4096 --gradient_checkpointing True --dataloader_num_workers 2
    --remove_unused_columns False --evaluation_strategy no --save_strategy no
    --logging_steps 1 --report_to wandb
    --lora_enable True --lora_llm True --lora_vt False
    --lora_r 64 --lora_alpha 128 --lora_dropout 0.05
    --tune_vision_tower False --tune_language_model True --tune_mm_projector True
    --tune_top_down_selection False --token_selection_loss_weight 0.0
    --train_w_gt_selection_map False --smooth_selection_prob_in_training False
    --ps3_grad_checkpointing True
)

printf 'Protocol: configs/train/matched312.json\nVariant: %s  Seed: %s\nCommand:\n  ' "$VARIANT" "$SEED"
printf '%q ' "${COMMAND[@]}"
printf '\n'

if [[ "$DRY_RUN" == true ]]; then
    exit 0
fi

export DATA_ROOT
export WANDB_MODE="${WANDB_MODE:-offline}"
cd "$VILA_ROOT"
exec "${COMMAND[@]}"
