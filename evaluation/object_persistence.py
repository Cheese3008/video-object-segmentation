import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CUTIE_DIR = Path(r"E:\NCKH-SV\experiments\results_cutie")
XMEM_DIR = Path(r"E:\NCKH-SV\experiments\results_xmem")
IMAGE_DIR = Path(r"E:\NCKH-SV\data\JPEGImages")
OUT_DIR = Path(r"E:\NCKH-SV\assets\Object Persistence")

# 🔥 CHỈ CHẠY 3 SEQUENCE
SELECTED_SEQS = ["0ql59q5s", "1wjebgyd", "3v2kgn6k"]

EXCLUDE_DIRS = {"experiments", "logs", "tmp", "__pycache__", "visual_cases"}

MIN_COMPONENT_AREA = 20
DISAPPEAR_AREA_THRESH = 50
SMOOTH_WINDOW = 5
NUM_VIS_CASES = 3

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# UTILS
# =========================
def sorted_pngs(seq_dir: Path):
    return sorted([p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])


def is_valid_sequence_dir(seq_dir: Path) -> bool:
    if not seq_dir.is_dir():
        return False
    if seq_dir.name in EXCLUDE_DIRS:
        return False
    return len(sorted_pngs(seq_dir)) > 0


def mask_to_binary(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("Mask is None")
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label] = 1
    return out


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

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out


def put_title(img: np.ndarray, text: str):
    out = img.copy()
    cv2.putText(
        out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 3, cv2.LINE_AA
    )
    cv2.putText(
        out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 0, 0), 1, cv2.LINE_AA
    )
    return out


# =========================
# ANALYSIS
# =========================
def analyze_sequence(model_dir: Path, model_name: str, seq_name: str) -> pd.DataFrame:
    seq_dir = model_dir / seq_name
    if not seq_dir.exists():
        print(f"[WARN] Không tìm thấy thư mục: {seq_dir}")
        return pd.DataFrame()

    frames = sorted_pngs(seq_dir)
    if len(frames) == 0:
        print(f"[WARN] Sequence {seq_name} không có file .png hợp lệ")
        return pd.DataFrame()

    rows = []

    for i, frame_path in enumerate(frames):
        raw = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        mask = mask_to_binary(raw)
        mask = remove_small_components(mask, MIN_COMPONENT_AREA)

        area = float(mask.sum())
        disappeared = int(area < DISAPPEAR_AREA_THRESH)

        rows.append({
            "model": model_name,
            "sequence": seq_name,
            "frame_idx": i,
            "frame_name": frame_path.name,
            "area": area,
            "disappeared": disappeared,
        })

    if len(rows) == 0:
        print(f"[WARN] Sequence {seq_name} của {model_name} không đọc được frame hợp lệ")
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =========================
# PLOTS
# =========================
def plot_area_curve(df_cutie, df_xmem, seq_name: str, save_path: Path):
    if df_cutie.empty or df_xmem.empty:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.2))

    x1 = df_cutie["frame_idx"].values
    y1 = smooth_series(df_cutie["area"].values, SMOOTH_WINDOW)

    x2 = df_xmem["frame_idx"].values
    y2 = smooth_series(df_xmem["area"].values, SMOOTH_WINDOW)

    plt.plot(x1, y1, label="CUTIE", linewidth=1.8)
    plt.plot(x2, y2, label="XMem", linewidth=1.8)
    plt.axhline(
        DISAPPEAR_AREA_THRESH,
        linestyle="--",
        linewidth=1.2,
        label="Disappear threshold"
    )

    plt.title(f"Mask Area (Smoothed) - Sequence {seq_name}")
    plt.xlabel("Frame index")
    plt.ylabel("Mask Area (pixels)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_disappearance_bar(summary_df: pd.DataFrame, save_path: Path):
    if summary_df.empty:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    pivot_df = summary_df.pivot(index="sequence", columns="model", values="disappearance_rate")

    ax = pivot_df.plot(kind="bar", figsize=(8, 4.5))
    ax.set_title("Disappearance Rate Comparison")
    ax.set_xlabel("Sequence")
    ax.set_ylabel("Disappearance Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_overall_persistence_bar(overall_df: pd.DataFrame, save_path: Path):
    if overall_df.empty:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    ax = overall_df.set_index("model")[["disappearance_rate", "persistence_ratio"]].plot(
        kind="bar", figsize=(6, 4)
    )
    ax.set_title("Overall Object Persistence Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# VISUAL CASES
# =========================
def make_persistence_case_panels(seq_name: str, df_cutie: pd.DataFrame, df_xmem: pd.DataFrame):
    if df_cutie.empty or df_xmem.empty:
        return

    case_dir = OUT_DIR / "visual_cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    merged = df_cutie.merge(
        df_xmem,
        on=["sequence", "frame_idx", "frame_name"],
        suffixes=("_cutie", "_xmem")
    )

    if merged.empty:
        return

    merged["area_gap"] = merged["area_cutie"] - merged["area_xmem"]
    merged["disappear_gap"] = merged["disappeared_xmem"] - merged["disappeared_cutie"]

    strong_cases = merged[merged["disappear_gap"] != 0].copy()

    if len(strong_cases) > 0:
        cases = strong_cases.sort_values(
            "area_gap", key=np.abs, ascending=False
        ).head(NUM_VIS_CASES)
    else:
        cases = merged.sort_values(
            "area_gap", key=np.abs, ascending=False
        ).head(NUM_VIS_CASES)

    for _, row in cases.iterrows():
        fname = row["frame_name"]

        cutie_mask_raw = cv2.imread(str(CUTIE_DIR / seq_name / fname), cv2.IMREAD_GRAYSCALE)
        xmem_mask_raw = cv2.imread(str(XMEM_DIR / seq_name / fname), cv2.IMREAD_GRAYSCALE)
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

        cutie_title = f"CUTIE area={int(row['area_cutie'])}"
        xmem_title = f"XMem area={int(row['area_xmem'])}"

        panel = np.hstack([
            put_title(image, f"Input: {seq_name}/{fname}"),
            put_title(cutie_overlay, cutie_title),
            put_title(xmem_overlay, xmem_title),
        ])

        out_name = f"{seq_name}_{fname.replace('.png', '')}.png"
        cv2.imwrite(str(case_dir / out_name), panel)


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Chỉ chạy các sequence: {SELECTED_SEQS}")

    all_rows = []
    summary_rows = []

    for idx, seq in enumerate(SELECTED_SEQS, start=1):
        print(f"[INFO] ({idx}/{len(SELECTED_SEQS)}) Đang xử lý: {seq}")

        if not is_valid_sequence_dir(CUTIE_DIR / seq):
            print(f"[SKIP] CUTIE không có sequence hợp lệ: {seq}")
            continue

        if not is_valid_sequence_dir(XMEM_DIR / seq):
            print(f"[SKIP] XMem không có sequence hợp lệ: {seq}")
            continue

        df_cutie = analyze_sequence(CUTIE_DIR, "CUTIE", seq)
        df_xmem = analyze_sequence(XMEM_DIR, "XMem", seq)

        if df_cutie.empty or df_xmem.empty:
            print(f"[SKIP] {seq} không có dữ liệu")
            continue

        all_rows.append(df_cutie)
        all_rows.append(df_xmem)

        # ===== summary =====
        for df_model in [df_cutie, df_xmem]:
            model_name = df_model["model"].iloc[0]

            disappearance_rate = float(df_model["disappeared"].mean())
            persistence_ratio = 1.0 - disappearance_rate

            summary_rows.append({
                "sequence": seq,
                "model": model_name,
                "mean_area": df_model["area"].mean(),
                "min_area": df_model["area"].min(),
                "max_area": df_model["area"].max(),
                "disappearance_rate": disappearance_rate,
                "persistence_ratio": persistence_ratio,
            })

        # ===== plot =====
        plot_area_curve(
            df_cutie,
            df_xmem,
            seq_name=seq,
            save_path=OUT_DIR / f"{seq}_area_persistence.png"
        )

        make_persistence_case_panels(seq, df_cutie, df_xmem)

    if len(all_rows) == 0:
        raise ValueError("Không có dữ liệu hợp lệ")

    all_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_df.to_csv(OUT_DIR / "metrics.csv", index=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    overall_df = summary_df.groupby("model", as_index=False).agg({
        "mean_area": "mean",
        "min_area": "mean",
        "max_area": "mean",
        "disappearance_rate": "mean",
        "persistence_ratio": "mean",
    })
    overall_df.to_csv(OUT_DIR / "summary_overall.csv", index=False)

    plot_disappearance_bar(
        summary_df,
        save_path=OUT_DIR / "summary_disappearance.png"
    )

    plot_overall_persistence_bar(
        overall_df,
        save_path=OUT_DIR / "summary_overall_persistence.png"
    )

    print("\n[OK] DONE - chỉ 3 sequence, dễ nhìn 👍")
    print(f"[OK] Đã lưu CSV chi tiết tại: {OUT_DIR / 'metrics.csv'}")
    print(f"[OK] Đã lưu CSV summary tại: {OUT_DIR / 'summary.csv'}")
    print(f"[OK] Đã lưu CSV overall tại: {OUT_DIR / 'summary_overall.csv'}")
    print(f"[OK] Đã lưu biểu đồ và ảnh trực quan tại: {OUT_DIR}")


if __name__ == "__main__":
    main()