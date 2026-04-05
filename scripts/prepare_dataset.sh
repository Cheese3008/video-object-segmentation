#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Script chuẩn bị dataset MOSEv2
# - Tải file valid.tar.gz
# - Giải nén về /workspace/datasets/valid
# - Chuẩn hóa dữ liệu cho XMem tại /workspace/datasets/mose_xmem
# =========================================

echo "[INFO] Checking GPU environment (optional)..."
nvidia-smi || true

# ===== Khai báo đường dẫn =====
WORKSPACE_DIR="/workspace"
DATASET_DIR="${WORKSPACE_DIR}/datasets"
MOSE_ARCHIVE="${DATASET_DIR}/valid.tar.gz"
MOSE_URL="https://huggingface.co/datasets/FudanCVL/MOSEv2/resolve/main/valid.tar.gz"

VALID_DIR="${DATASET_DIR}/valid"
JPEG_DIR="${VALID_DIR}/JPEGImages"
ANNOTATION_DIR="${VALID_DIR}/Annotations"

XMEM_DATASET_DIR="${DATASET_DIR}/mose_xmem"

# ===== Tạo thư mục datasets =====
mkdir -p "${DATASET_DIR}"

echo "[INFO] Workspace directory: ${WORKSPACE_DIR}"
echo "[INFO] Dataset directory: ${DATASET_DIR}"

# ===== Bước 1: tải dataset MOSE =====
if [ -f "${MOSE_ARCHIVE}" ]; then
    echo "[INFO] Archive already exists: ${MOSE_ARCHIVE}"
else
    echo "[INFO] Downloading MOSEv2 valid split..."
    wget -c "${MOSE_URL}" -O "${MOSE_ARCHIVE}"
fi

# ===== Bước 2: giải nén =====
if [ -d "${VALID_DIR}" ]; then
    echo "[INFO] Extracted dataset already exists: ${VALID_DIR}"
else
    echo "[INFO] Extracting dataset..."
    tar -xzf "${MOSE_ARCHIVE}" -C "${DATASET_DIR}"
fi

# ===== Kiểm tra cấu trúc dữ liệu gốc =====
if [ ! -d "${JPEG_DIR}" ]; then
    echo "[ERROR] JPEGImages directory not found: ${JPEG_DIR}"
    exit 1
fi

if [ ! -d "${ANNOTATION_DIR}" ]; then
    echo "[ERROR] Annotations directory not found: ${ANNOTATION_DIR}"
    exit 1
fi

echo "[INFO] Sample sequences:"
ls "${JPEG_DIR}" | head

# ===== Bước 3: chuẩn hóa dữ liệu cho XMem =====
echo "[INFO] Preparing MOSE dataset format for XMem..."

rm -rf "${XMEM_DATASET_DIR}"
mkdir -p "${XMEM_DATASET_DIR}"

python3 - <<'PY'
from pathlib import Path
import os
import shutil

workspace_dir = Path("/workspace")
src_img = workspace_dir / "datasets" / "valid" / "JPEGImages"
src_mask = workspace_dir / "datasets" / "valid" / "Annotations"
dst = workspace_dir / "datasets" / "mose_xmem"

def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

seqs = sorted([p for p in src_img.iterdir() if p.is_dir()])
print(f"[INFO] Found {len(seqs)} sequences")

for seq in seqs:
    name = seq.name
    dst_seq = dst / name
    dst_seq.mkdir(parents=True, exist_ok=True)

    image_files = sorted(seq.glob("*.jpg"))
    for img in image_files:
        safe_symlink(img, dst_seq / img.name)

    mask0 = src_mask / name / "00000.png"
    if mask0.exists():
        safe_symlink(mask0, dst_seq / "00000.png")

    print(f"[OK] Prepared sequence: {name} | frames={len(image_files)}")

print("[OK] MOSE dataset prepared for XMem.")
PY

# ===== Kiểm tra output =====
echo "[INFO] Checking XMem dataset structure..."
find "${XMEM_DATASET_DIR}" -maxdepth 2 | head -30

echo "[OK] Dataset preparation completed."
echo "[OK] Original MOSE: ${VALID_DIR}"
echo "[OK] XMem-ready MOSE: ${XMEM_DATASET_DIR}"