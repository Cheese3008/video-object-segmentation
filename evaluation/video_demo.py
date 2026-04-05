import cv2
import numpy as np
from pathlib import Path

sequence_name = "0ql59q5s"

rgb_dir = Path(r"E:\NCKH-SV\data\JPEGImages") / sequence_name
cutie_dir = Path(r"E:\NCKH-SV\experiments\results_cutie") / sequence_name
xmem_dir = Path(r"E:\NCKH-SV\experiments\results_xmem") / sequence_name

output_video = str(Path(r"E:\NCKH-SV\assets\results") / f"{sequence_name}_compare.mp4")
fps = 8
alpha = 0.45

def get_image_files(folder: Path):
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [f for f in folder.iterdir() if f.suffix.lower() in valid_exts]
    return sorted(files, key=lambda x: x.name)

def overlay_mask_on_image(image, mask, color=(0, 0, 255), alpha=0.45):
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    binary_mask = mask_gray > 0
    overlay = image.copy()
    overlay[binary_mask] = (
        (1 - alpha) * overlay[binary_mask] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay

def make_overlay_compare_video(rgb_files, cutie_files, xmem_files, output_path, fps=10):
    num_frames = min(len(rgb_files), len(cutie_files), len(xmem_files))
    if num_frames == 0:
        print("[ERROR] Khong du du lieu de tao video")
        return

    first_img = cv2.imread(str(rgb_files[0]))
    h, w = first_img.shape[:2]
    top_bar_h = 60
    final_w = w * 2
    final_h = h + top_bar_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (final_w, final_h))

    for i in range(num_frames):
        rgb = cv2.imread(str(rgb_files[i]))
        cutie_mask = cv2.imread(str(cutie_files[i]), cv2.IMREAD_UNCHANGED)
        xmem_mask = cv2.imread(str(xmem_files[i]), cv2.IMREAD_UNCHANGED)

        if rgb is None or cutie_mask is None or xmem_mask is None:
            continue

        rgb = cv2.resize(rgb, (w, h))
        cutie_mask = cv2.resize(cutie_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        xmem_mask = cv2.resize(xmem_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        cutie_overlay = overlay_mask_on_image(rgb.copy(), cutie_mask, color=(0, 0, 255), alpha=alpha)
        xmem_overlay = overlay_mask_on_image(rgb.copy(), xmem_mask, color=(0, 255, 255), alpha=alpha)

        combined = cv2.hconcat([cutie_overlay, xmem_overlay])
        canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        canvas[top_bar_h:, :] = combined

        cv2.putText(canvas, f"Sequence: {sequence_name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, "XMem", (w // 2 - 50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(canvas, "CUTIE", (w + w // 2 - 40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.line(canvas, (w, top_bar_h), (w, final_h), (255, 255, 255), 2)

        writer.write(canvas)

    writer.release()
    print(f"[OK] Da tao video: {output_path}")

rgb_files = get_image_files(rgb_dir)
cutie_files = get_image_files(cutie_dir)
xmem_files = get_image_files(xmem_dir)

print(f"RGB   : {len(rgb_files)}")
print(f"CUTIE : {len(cutie_files)}")
print(f"XMem  : {len(xmem_files)}")

make_overlay_compare_video(rgb_files, cutie_files, xmem_files, output_video, fps=fps)