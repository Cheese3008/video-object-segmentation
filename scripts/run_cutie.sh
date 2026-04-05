#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Script chạy CUTIE trên server GPU
# Phù hợp với workspace RunPod của project này
# =========================================

echo "[INFO] Checking GPU..."
nvidia-smi

# ===== Khai báo đường dẫn =====
WORKSPACE_DIR="/workspace"
CUTIE_ENV="${WORKSPACE_DIR}/cutie_env"
CUTIE_DIR="${WORKSPACE_DIR}/Cutie"
WEIGHTS_PATH="${CUTIE_DIR}/weights/cutie-base-mega.pth"
IMAGE_DIR="${WORKSPACE_DIR}/datasets/valid/JPEGImages"
MASK_DIR="${WORKSPACE_DIR}/datasets/valid/Annotations"
OUTPUT_DIR="${WORKSPACE_DIR}/experiments/results_cutie"

# ===== Kiểm tra tồn tại =====
if [ ! -d "${CUTIE_DIR}" ]; then
    echo "[ERROR] CUTIE_DIR not found: ${CUTIE_DIR}"
    exit 1
fi

if [ ! -d "${CUTIE_ENV}" ]; then
    echo "[ERROR] CUTIE_ENV not found: ${CUTIE_ENV}"
    exit 1
fi

if [ ! -f "${WEIGHTS_PATH}" ]; then
    echo "[ERROR] Weights not found: ${WEIGHTS_PATH}"
    exit 1
fi

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "[ERROR] IMAGE_DIR not found: ${IMAGE_DIR}"
    exit 1
fi

if [ ! -d "${MASK_DIR}" ]; then
    echo "[ERROR] MASK_DIR not found: ${MASK_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ===== Activate môi trường =====
echo "[INFO] Activating CUTIE environment..."
source "${CUTIE_ENV}/bin/activate"

echo "[INFO] Checking PyTorch CUDA..."
python -c "import torch; print('torch.cuda.is_available =', torch.cuda.is_available()); print('torch.cuda.device_count =', torch.cuda.device_count())"

# ===== Chạy CUTIE =====
cd "${CUTIE_DIR}"

echo "[INFO] Running CUTIE inference..."
python -m cutie.eval_vos \
    dataset=generic \
    amp=true \
    weights="${WEIGHTS_PATH}" \
    output_dir="${OUTPUT_DIR}" \
    image_directory="${IMAGE_DIR}" \
    mask_directory="${MASK_DIR}" \
    size=480 \
    save_all=true \
    use_all_masks=false \
    hydra.run.dir=.

echo "[OK] CUTIE inference done."
echo "[OK] Output saved at: ${OUTPUT_DIR}"

echo "[INFO] Sample output structure:"
find "${OUTPUT_DIR}" -maxdepth 3 | head -30