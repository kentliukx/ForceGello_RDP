#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

GPU_ID=${GPU_ID:-0}
DATASET_PATH=${DATASET_PATH:-data/forcegello_zarr}
LOGGING_MODE=${LOGGING_MODE:-offline}
RUN_DIR=${RUN_DIR:-"$(pwd)/data/outputs/forcegello_right_$(date +%Y%m%d_%H%M%S)"}

if [[ ! -d "${DATASET_PATH}/replay_buffer.zarr" ]]; then
    echo "Missing ${DATASET_PATH}/replay_buffer.zarr; run post_process_forcegello.py first." >&2
    exit 1
fi

echo "Stage 1: right-arm Asymmetric Tokenizer"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
    --config-name=train_forcegello_right_at \
    "task.dataset_path=${DATASET_PATH}" \
    "hydra.run.dir=${RUN_DIR}/at" \
    "logging.mode=${LOGGING_MODE}"

AT_LOAD_DIR="${RUN_DIR}/at/checkpoints/latest.ckpt"
if [[ ! -f "${AT_LOAD_DIR}" ]]; then
    echo "AT checkpoint missing: ${AT_LOAD_DIR}" >&2
    exit 1
fi

echo "Stage 2: right-arm Latent Diffusion Policy"
CUDA_VISIBLE_DEVICES="${GPU_ID}" accelerate launch --num_processes=1 train.py \
    --config-name=train_forcegello_right_rdp \
    "task.dataset_path=${DATASET_PATH}" \
    "at_load_dir=${AT_LOAD_DIR}" \
    "hydra.run.dir=${RUN_DIR}/rdp" \
    "logging.mode=${LOGGING_MODE}"
