r"""detect_shakiness.py -- per-clip handheld-camera jitter score, filling a real gap:
bootlegcam's own build_song.py already reads an optional `shakiness-report.csv`
(steadiness_score()/load_shakiness_map()) if one happens to exist beside the Masters folder,
but nothing in that skill ever generates it -- this does.

Metric (column `jitter_std_px`, the exact name build_song.py expects): sample frames at a low
fixed rate, estimate frame-to-frame global motion via phase correlation (cv2.phaseCorrelate --
robust to scene content, standard technique for this), then take the STANDARD DEVIATION of that
displacement across the clip -- not its mean. A smooth intentional pan has large but CONSISTENT
displacement (low std); handheld jitter has small but ERRATIC, inconsistent displacement
(high std) -- std of frame-to-frame motion separates "camera is moving" from "camera is shaking"
in a way a raw motion-magnitude average cannot.

Usage:
  py detect_shakiness.py <masters_dir> <out_csv> [--sample-fps 5]
"""
import csv
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

from clip_utils import find_clips


def probe_wh(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=,", path],
                        capture_output=True, text=True)
    w, h = r.stdout.strip().rstrip(",").split(",")
    return int(w), int(h)


def decode_frames_small(path, sample_fps, out_w=480):
    """Yield successive grayscale frames at sample_fps, pre-scaled to out_w, via NVDEC+fps --
    same decode pattern as extract_people.py's decode_frames(), not a CPU cv2.VideoCapture read
    of every source frame just to throw most of them away. On a long 4K60 GoPro/phone clip, the
    old approach decoded every frame on CPU and discarded ~11 of every 12 (sample_fps=5 vs
    60fps) in Python; here ffmpeg's own fps filter does the frame selection on the GPU decoder."""
    src_w, src_h = probe_wh(path)
    out_h = max(2, round(out_w * src_h / src_w / 2) * 2)  # even height, same rounding scale=-2 uses
    dec = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-v", "error", "-hwaccel", "cuda", "-i", path,
         "-vf", f"fps={sample_fps},scale={out_w}:{out_h},format=gray", "-f", "rawvideo",
         "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, bufsize=out_w * out_h * 4)
    fsz = out_w * out_h
    while True:
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz:
            break
        yield np.frombuffer(buf, np.uint8).reshape(out_h, out_w).copy()
    dec.wait()


def jitter_for_clip(path, sample_fps):
    prev_gray = None
    dx_list, dy_list = [], []
    for gray in decode_frames_small(path, sample_fps):
        gray = np.float32(gray)
        if prev_gray is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev_gray, gray)
            dx_list.append(dx)
            dy_list.append(dy)
        prev_gray = gray

    if len(dx_list) < 5:
        return None
    jitter = float(np.hypot(np.std(dx_list), np.std(dy_list)))
    return jitter


def main():
    masters_dir, out_csv = sys.argv[1], sys.argv[2]
    sample_fps = 5.0
    if "--sample-fps" in sys.argv:
        sample_fps = float(sys.argv[sys.argv.index("--sample-fps") + 1])

    clips = find_clips(masters_dir)
    rows = []
    for i, path in enumerate(clips):
        jitter = jitter_for_clip(path, sample_fps)
        status = f"{jitter:.2f}px" if jitter is not None else "FAILED"
        print(f"[{i+1}/{len(clips)}] {os.path.basename(path)}: {status}")
        if jitter is not None:
            rows.append((os.path.basename(path), jitter))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "jitter_std_px"])
        for file, jitter in rows:
            w.writerow([file, f"{jitter:.3f}"])
    print(f"wrote {out_csv} ({len(rows)} clips)")


if __name__ == "__main__":
    main()
