import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CUTIE_DIR = Path(r"E:\NCKH-SV\results_cutie")
XMEM_DIR = Path(r"E:\NCKH-SV\results_xmem")
OUT_DIR = Path(r"E:\NCKH-SV\assets\Temporal Stability")

# Chọn sequence muốn vẽ. Nếu để None thì vẽ tất cả sequence chung giữa 2 model
SELECTED_SEQS = ["0ql59q5s", "1wjebgyd", "3v2kgn6k"]
# SELECTED_SEQS = None

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# UTILS
# =========================
def sorted_pngs(seq_dir: Path):
    return sorted([p for p in seq_dir.iterdir() if p.suffix.lower() == ".png"])


def mask_to_binary(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("Mask is None")
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8)


def temporal_iou(mask_prev: np.ndarray, mask_curr: np.ndarray) -> float:
    inter = np.logical_and(mask_prev > 0, mask_curr > 0).sum()
    union = np.logical_or(mask_prev > 0, mask_curr > 0).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def centroid(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.nan, np.nan
    return float(xs.mean()), float(ys.mean())


def centroid_shift(mask_prev: np.ndarray, mask_curr: np.ndarray) -> float:
    x1, y1 = centroid(mask_prev)
    x2, y2 = centroid(mask_curr)

    if np.isnan(x1) or np.isnan(x2):
        return np.nan

    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def analyze_sequence(model_dir: Path, model_name: str, seq_name: str) -> pd.DataFrame:
    seq_dir = model_dir / seq_name
    if not seq_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {seq_dir}")

    frames = sorted_pngs(seq_dir)
    if len(frames) < 2:
        raise ValueError(f"Sequence {seq_name} không đủ frame để tính temporal metrics")

    rows = []
    prev_mask = None

    for i, frame_path in enumerate(frames):
        raw = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        curr_mask = mask_to_binary(raw)

        if prev_mask is not None:
            rows.append({
                "model": model_name,
                "sequence": seq_name,
                "frame_idx": i,
                "frame_name": frame_path.name,
                "temporal_iou": temporal_iou(prev_mask, curr_mask),
                "centroid_shift": centroid_shift(prev_mask, curr_mask),
            })

        prev_mask = curr_mask

    return pd.DataFrame(rows)


def smooth_series(values, window=5):
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1).mean().values


def plot_one_metric(
    df_cutie: pd.DataFrame,
    df_xmem: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    save_path: Path,
    smooth: bool = True
):
    plt.figure(figsize=(9, 4.2))

    x1 = df_cutie["frame_idx"].values
    y1 = df_cutie[metric_col].values
    if smooth:
        y1 = smooth_series(y1, window=5)

    x2 = df_xmem["frame_idx"].values
    y2 = df_xmem[metric_col].values
    if smooth:
        y2 = smooth_series(y2, window=5)

    plt.plot(x1, y1, label="CUTIE", linewidth=1.8)
    plt.plot(x2, y2, label="XMem", linewidth=1.8)

    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    cutie_seqs = {p.name for p in CUTIE_DIR.iterdir() if p.is_dir()}
    xmem_seqs = {p.name for p in XMEM_DIR.iterdir() if p.is_dir()}
    common_seqs = sorted(list(cutie_seqs & xmem_seqs))

    if SELECTED_SEQS is not None:
        common_seqs = [s for s in common_seqs if s in SELECTED_SEQS]

    if not common_seqs:
        raise ValueError("Không có sequence chung giữa CUTIE và XMem")

    all_rows = []

    for seq in common_seqs:
        print(f"[INFO] Đang xử lý sequence: {seq}")

        df_cutie = analyze_sequence(CUTIE_DIR, "CUTIE", seq)
        df_xmem = analyze_sequence(XMEM_DIR, "XMem", seq)

        all_rows.append(df_cutie)
        all_rows.append(df_xmem)

        # 1) Temporal IoU (SMOOTH ONLY)
        plot_one_metric(
            df_cutie,
            df_xmem,
            metric_col="temporal_iou",
            ylabel="Temporal IoU",
            title=f"Temporal IoU (Smoothed) - Sequence {seq}",
            save_path=OUT_DIR / f"{seq}_temporal_iou.png",
            smooth=True
        )

        # 2) Centroid Shift (SMOOTH ONLY)
        plot_one_metric(
            df_cutie,
            df_xmem,
            metric_col="centroid_shift",
            ylabel="Centroid Shift (pixels)",
            title=f"Centroid Shift (Smoothed) - Sequence {seq}",
            save_path=OUT_DIR / f"{seq}_centroid_shift.png",
            smooth=True
        )

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "temporal_stability_metrics.csv", index=False)

    print(f"\n[OK] Đã lưu metric CSV tại: {OUT_DIR / 'temporal_stability_metrics.csv'}")
    print(f"[OK] Đã lưu biểu đồ (2 mỗi sequence) tại: {OUT_DIR}")


if __name__ == "__main__":
    main()