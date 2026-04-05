#Biểu đồ thể hiện thời gian xử lý theo từng frame (frame-wise latency) được sử dụng để phân tích sự thay đổi độ trễ theo thời gian.#

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATH = r"E:\NCKH-SV\assets\Efficiency\frame_time_comparison.csv"
SEQUENCE_NAME = "0ql59q5s"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

# lọc theo sequence
df_seq = df[df["sequence"] == SEQUENCE_NAME].copy()

if df_seq.empty:
    raise ValueError(f"Không tìm thấy sequence: {SEQUENCE_NAME}")

# =========================
# TÁCH 2 MODEL
# =========================
cutie_df = df_seq[df_seq["model"] == "CUTIE"].copy()
xmem_df = df_seq[df_seq["model"] == "XMem"].copy()

# =========================
# PLOT (2 LINE TRÊN 1 BIỂU ĐỒ)
# =========================
plt.figure(figsize=(9, 4))

# CUTIE
plt.plot(
    cutie_df["frame_idx"],
    cutie_df["time_ms"],
    marker="o",
    label="CUTIE"
)

# XMem
plt.plot(
    xmem_df["frame_idx"],
    xmem_df["time_ms"],
    marker="s",
    label="XMem"
)

# =========================
# DECORATION
# =========================
plt.title(f"Latency Comparison - Sequence {SEQUENCE_NAME}")
plt.xlabel("Frame index")
plt.ylabel("Time (ms)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()