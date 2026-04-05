#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Script chạy XMem trên server GPU
# Phù hợp với workspace RunPod của project này
# =========================================

echo "[INFO] Checking GPU..."
nvidia-smi

# ===== Khai báo đường dẫn =====
WORKSPACE_DIR="/workspace"
XMEM_ENV="${WORKSPACE_DIR}/xmem_env"
XMEM_DIR="${WORKSPACE_DIR}/XMem"
MODEL_PATH="${XMEM_DIR}/saves/XMem.pth"
DATASET_PATH="${WORKSPACE_DIR}/datasets/valid"
OUTPUT_DIR="${WORKSPACE_DIR}/experiments/results_xmem"

# ===== Kiểm tra tồn tại =====
if [ ! -d "${XMEM_DIR}" ]; then
    echo "[ERROR] XMEM_DIR not found: ${XMEM_DIR}"
    exit 1
fi

if [ ! -d "${XMEM_ENV}" ]; then
    echo "[ERROR] XMEM_ENV not found: ${XMEM_ENV}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "[ERROR] Model not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -d "${DATASET_PATH}" ]; then
    echo "[ERROR] DATASET_PATH not found: ${DATASET_PATH}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ===== Activate môi trường =====
echo "[INFO] Activating XMem environment..."
source "${XMEM_ENV}/bin/activate"

echo "[INFO] Checking PyTorch CUDA..."
python -c "import torch; print('torch.cuda.is_available =', torch.cuda.is_available()); print('torch.cuda.device_count =', torch.cuda.device_count())"

# ===== Chạy XMem =====
cd "${XMEM_DIR}"

echo "[INFO] Running XMem inference..."
python eval.py \
    --model "${MODEL_PATH}" \
    --generic_path "${DATASET_PATH}" \
    --dataset G \
    --output "${OUTPUT_DIR}"

echo "[OK] XMem inference done."
echo "[OK] Output saved at: ${OUTPUT_DIR}"

echo "[INFO] Sample output structure:"
find "${OUTPUT_DIR}" -maxdepth 3 | head -30