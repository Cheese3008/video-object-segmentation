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
IMAGE_DIR = Path(r"E:\NCKH-SV\JPEGImages")  
OUT_DIR = Path(r"E:\NCKH-SV\assets\Shape Quality")  

SELECTED_SEQS = ["0ql59q5s", "1wjebgyd", "3v2kgn6k"]
# SELECTED_SEQS = None

MIN_COMPONENT_AREA = 20
NUM_VIS_FRAMES = 3
SMOOTH_WINDOW = 5
EPS = 1e-9

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


def remove_small_components(mask: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label] = 1
    return out


def connected_components_count(mask: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> int:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    count = 0
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            count += 1
    return count


def boundary_roughness(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area < 1:
        return np.nan

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.nan

    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return float((perimeter ** 2) / (4.0 * np.pi * area + EPS))


def smooth_series(values, window=5):
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1).mean().values


def load_image_for_frame(seq_name: str, png_name: str):
    jpg_name = png_name.replace(".png", ".jpg")
    img_path = IMAGE_DIR / seq_name / jpg_name
    if not img_path.exists():
        return None
    return cv2.imread(str(img_path))


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, color, alpha=0.45):
    out = image.copy()
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = color
    out = cv2.addWeighted(out, 1.0, color_mask, alpha, 0)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out


def put_title(img: np.ndarray, text: str):
    out = img.copy()
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# =========================
# ANALYSIS
# =========================
def analyze_sequence(model_dir: Path, model_name: str, seq_name: str) -> pd.DataFrame:
    seq_dir = model_dir / seq_name
    if not seq_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {seq_dir}")

    frames = sorted_pngs(seq_dir)
    rows = []

    for i, frame_path in enumerate(frames):
        raw = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        mask = mask_to_binary(raw)
        mask = remove_small_components(mask, MIN_COMPONENT_AREA)

        area = float(mask.sum())
        fragmentation = connected_components_count(mask, MIN_COMPONENT_AREA)
        roughness = boundary_roughness(mask)

        rows.append({
            "model": model_name,
            "sequence": seq_name,
            "frame_idx": i,
            "frame_name": frame_path.name,
            "area": area,
            "fragmentation": fragmentation,
            "boundary_roughness": roughness,
        })

    return pd.DataFrame(rows)


# =========================
# PLOTS
# =========================
def plot_metric(df_cutie, df_xmem, metric_col, ylabel, title, save_path):
    plt.figure(figsize=(9, 4.2))

    x1 = df_cutie["frame_idx"].values
    y1 = smooth_series(df_cutie[metric_col].values, SMOOTH_WINDOW)

    x2 = df_xmem["frame_idx"].values
    y2 = smooth_series(df_xmem[metric_col].values, SMOOTH_WINDOW)

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


def make_summary_bar(summary_df: pd.DataFrame, metric_col: str, ylabel: str, title: str, save_path: Path):
    pivot_df = summary_df.pivot(index="sequence", columns="model", values=metric_col)

    plt.figure(figsize=(8, 4.5))
    pivot_df.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Sequence")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# VISUAL PANELS
# =========================
def make_visual_panels(seq_name: str):
    cutie_seq = CUTIE_DIR / seq_name
    xmem_seq = XMEM_DIR / seq_name

    cutie_frames = sorted_pngs(cutie_seq)
    xmem_frames = sorted_pngs(xmem_seq)
    common_names = sorted(set(p.name for p in cutie_frames) & set(p.name for p in xmem_frames))

    if len(common_names) == 0:
        return

    idxs = np.linspace(0, len(common_names) - 1, NUM_VIS_FRAMES, dtype=int)
    vis_dir = OUT_DIR / "visual_panels"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for idx in idxs:
        fname = common_names[idx]

        cutie_mask_raw = cv2.imread(str(cutie_seq / fname), cv2.IMREAD_GRAYSCALE)
        xmem_mask_raw = cv2.imread(str(xmem_seq / fname), cv2.IMREAD_GRAYSCALE)
        if cutie_mask_raw is None or xmem_mask_raw is None:
            continue

        cutie_mask = remove_small_components(mask_to_binary(cutie_mask_raw))
        xmem_mask = remove_small_components(mask_to_binary(xmem_mask_raw))

        image = load_image_for_frame(seq_name, fname)
        if image is None:
            h, w = cutie_mask.shape
            image = np.full((h, w, 3), 180, dtype=np.uint8)

        cutie_overlay = overlay_mask_on_image(image, cutie_mask, color=(0, 255, 0))
        xmem_overlay = overlay_mask_on_image(image, xmem_mask, color=(0, 165, 255))

        panel = np.hstack([
            put_title(image, f"Input: {seq_name}/{fname}"),
            put_title(cutie_overlay, "CUTIE"),
            put_title(xmem_overlay, "XMem")
        ])

        cv2.imwrite(str(vis_dir / f"{seq_name}_{fname.replace('.png', '')}.png"), panel)


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
    summary_rows = []

    for seq in common_seqs:
        print(f"[INFO] Đang xử lý sequence: {seq}")

        df_cutie = analyze_sequence(CUTIE_DIR, "CUTIE", seq)
        df_xmem = analyze_sequence(XMEM_DIR, "XMem", seq)

        all_rows.append(df_cutie)
        all_rows.append(df_xmem)

        # 1) Boundary Roughness
        plot_metric(
            df_cutie, df_xmem,
            metric_col="boundary_roughness",
            ylabel="Boundary Roughness",
            title=f"Boundary Roughness (Smoothed) - Sequence {seq}",
            save_path=OUT_DIR / f"{seq}_boundary_roughness.png"
        )

        # 2) Fragmentation
        plot_metric(
            df_cutie, df_xmem,
            metric_col="fragmentation",
            ylabel="Fragmentation",
            title=f"Fragmentation (Smoothed) - Sequence {seq}",
            save_path=OUT_DIR / f"{seq}_fragmentation.png"
        )

        # 3) Area
        plot_metric(
            df_cutie, df_xmem,
            metric_col="area",
            ylabel="Area (pixels)",
            title=f"Mask Area (Smoothed) - Sequence {seq}",
            save_path=OUT_DIR / f"{seq}_area.png"
        )

        # summary per model/sequence
        for df_model in [df_cutie, df_xmem]:
            model_name = df_model["model"].iloc[0]

            area_mean = df_model["area"].mean()
            area_std = df_model["area"].std()
            area_cv = area_std / (area_mean + EPS)

            summary_rows.append({
                "sequence": seq,
                "model": model_name,
                "mean_boundary_roughness": df_model["boundary_roughness"].mean(),
                "mean_fragmentation": df_model["fragmentation"].mean(),
                "mean_area": area_mean,
                "area_cv": area_cv,
            })

        # visual panels
        make_visual_panels(seq)

    all_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_df.to_csv(OUT_DIR / "shape_quality_metrics.csv", index=False)
    summary_df.to_csv(OUT_DIR / "shape_quality_summary.csv", index=False)

    # summary bars
    make_summary_bar(
        summary_df,
        metric_col="mean_boundary_roughness",
        ylabel="Mean Boundary Roughness",
        title="Boundary Roughness Comparison",
        save_path=OUT_DIR / "summary_boundary_roughness.png"
    )

    make_summary_bar(
        summary_df,
        metric_col="mean_fragmentation",
        ylabel="Mean Fragmentation",
        title="Fragmentation Comparison",
        save_path=OUT_DIR / "summary_fragmentation.png"
    )

    make_summary_bar(
        summary_df,
        metric_col="area_cv",
        ylabel="Area CV",
        title="Area Stability Comparison",
        save_path=OUT_DIR / "summary_area_cv.png"
    )

    print(f"\n[OK] Đã lưu CSV chi tiết tại: {OUT_DIR / 'shape_quality_metrics.csv'}")
    print(f"[OK] Đã lưu CSV summary tại: {OUT_DIR / 'shape_quality_summary.csv'}")
    print(f"[OK] Đã lưu biểu đồ và ảnh trực quan tại: {OUT_DIR}")


if __name__ == "__main__":
    main()