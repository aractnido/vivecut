r"""apply_lens_correction_gpu.py -- GPU-accelerated drop-in replacement for
apply_lens_correction.py. Same k1/k2/rotation/crop math (reuses build_remap, crop_box,
apply_tone from lens_calibrator.py so results match the GUI-approved look), but the
undistort + rotate remaps run as torch CUDA grid_sample calls instead of CPU cv2.remap/
warpAffine, and the final encode uses NVENC (h264_nvenc) instead of libx264. Decode stays
on CPU (cv2.VideoCapture) -- an NVDEC decode path was tried and measured SLOWER end-to-end
than this (see "NVDEC was tried and reverted" below), so don't re-add it without re-proving
it actually helps on a timed before/after, not just in theory.

A single clip's pipeline barely touches the GPU on its own (nvidia-smi dmon showed low
single-digit % on the sm/encoder engines running one clip at a time) -- the per-frame
transform is sub-millisecond on a 4090 and NVENC can keep up with far more than one
clip's worth of frames. --parallel N runs N clips' pipelines concurrently (threads; cv2
decode releases the GIL during the actual decode call, and torch CUDA ops release it
during kernel execution, so this gets real overlap, not just interleaving) instead of
leaving the GPU mostly idle between batches of a single clip.

NVDEC was tried and reverted: decoding via `ffmpeg -hwaccel cuda` alone is fast in
isolation (~36fps raw for this footage), but wiring it into this script via a background
reader thread + queue made the full pipeline SLOWER than plain cv2 decode (3m15s+ vs 56s
for the same 30s demo clip, before being killed for taking too long). The likely cause:
each raw 4K frame is ~25MB, and shuttling that through Python twice per frame (once from
the NVDEC pipe into a queue, once again into the NVENC pipe) costs more in Python/GIL/
memory-copy overhead than NVDEC saves over cv2's own (already fairly fast) decode. A
proper fix would mean keeping frames in GPU memory end-to-end (e.g. via NVIDIA's VPF /
PyNvVideoCodec, not plain ffmpeg subprocess pipes) -- a bigger lift, not done here.

Why two separate grid_sample passes (undistort, then rotate) instead of composing one
combined sampling grid: this mirrors apply_transform()'s own two-step cv2.remap +
cv2.warpAffine structure exactly, so the GPU path reproduces the same output characteristics
(including the same double-interpolation softening) as the CPU path that was already
visually approved on a demo render, rather than silently producing a sharper-but-different
single-pass result.

Usage: identical to apply_lens_correction.py, plus --parallel --
  py apply_lens_correction_gpu.py <src_dir> <glob> <out_dir> k1 k2 angle_deg crop_top crop_right crop_bottom crop_left
  py apply_lens_correction_gpu.py ... --demo <start_sec> <duration_sec>
  py apply_lens_correction_gpu.py ... --shadow <0.0-1.0> --brightness <-1.0-1.0> --blacks <-1.0-1.0>
  py apply_lens_correction_gpu.py ... --batch <N>      # frames per GPU batch, default 16
  py apply_lens_correction_gpu.py ... --parallel <N>   # clips processed concurrently, default 1
"""
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lens_calibrator import apply_tone, build_remap, crop_box

DEVICE = torch.device("cuda")


def undistort_grid(w, h, k1, k2):
    map_x, map_y = build_remap(w, h, k1, k2)  # pixel-space source coords, shape (h, w)
    gx = (map_x / (w - 1)) * 2.0 - 1.0
    gy = (map_y / (h - 1)) * 2.0 - 1.0
    grid = np.stack([gx, gy], axis=-1).astype(np.float32)
    return torch.from_numpy(grid).unsqueeze(0).to(DEVICE)  # (1, h, w, 2)


def rotation_grid(w, h, angle_deg):
    # Same convention as apply_transform(): forward matrix via getRotationMatrix2D(-angle),
    # then invert it, since that's what cv2.warpAffine does internally without WARP_INVERSE_MAP.
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)
    Minv = cv2.invertAffineTransform(M)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
    src_y = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
    gx = (src_x / (w - 1)) * 2.0 - 1.0
    gy = (src_y / (h - 1)) * 2.0 - 1.0
    grid = np.stack([gx, gy], axis=-1).astype(np.float32)
    return torch.from_numpy(grid).unsqueeze(0).to(DEVICE)


def process_batch(frames_bgr, undist_grid, rot_grid):
    """frames_bgr: list of HxWx3 uint8 numpy arrays (already tone-adjusted). Returns the same
    shape, geometry-corrected, as a list of uint8 numpy arrays."""
    n = len(frames_bgr)
    batch = torch.from_numpy(np.stack(frames_bgr)).to(DEVICE)  # (N, H, W, 3) uint8
    batch = batch.permute(0, 3, 1, 2).float()  # (N, 3, H, W)

    g1 = undist_grid.expand(n, -1, -1, -1)
    undistorted = F.grid_sample(batch, g1, mode="bilinear", padding_mode="zeros", align_corners=True)

    g2 = rot_grid.expand(n, -1, -1, -1)
    rotated = F.grid_sample(undistorted, g2, mode="bilinear", padding_mode="zeros", align_corners=True)

    rotated = rotated.clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return [rotated[i] for i in range(n)]


def process_one(src, out_path, k1, k2, angle, ct, cr, cb, cl,
                 shadow=0.0, brightness=0.0, blacks=0.0,
                 start_sec=0.0, duration_sec=None, batch_size=16):
    tag = os.path.basename(src)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_ratio = w / h
    x0, y0, x1, y1 = crop_box(w, h, ct, cr, cb, cl, target_ratio)
    out_w, out_h = x1 - x0, y1 - y0
    out_w -= out_w % 2
    out_h -= out_h % 2
    x1, y1 = x0 + out_w, y0 + out_h

    total_dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0
    dur = duration_sec if duration_sec is not None else max(0.0, total_dur - start_sec)
    n_frames = int(round(dur * fps))
    print(f"[{tag}] {w}x{h} @ {fps:.2f}fps -> {out_w}x{out_h}, "
          f"{start_sec:.1f}s + {dur:.1f}s ({n_frames} frames) [GPU transform + NVENC]")

    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    u_grid = undistort_grid(w, h, k1, k2)
    r_grid = rotation_grid(w, h, angle)

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.5f}", "-i", "-",
        "-ss", str(start_sec), "-t", str(dur), "-i", src,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k", "-shortest", out_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    done = 0
    pending = []
    while done < n_frames:
        ok, frame = cap.read()
        if not ok:
            print(f"[{tag}] stopped early at frame {done}/{n_frames} (end of file)")
            break
        frame = apply_tone(frame, shadow, brightness, blacks)
        pending.append(frame)
        if len(pending) == batch_size or done + len(pending) == n_frames:
            out_frames = process_batch(pending, u_grid, r_grid)
            for out in out_frames:
                out = out[y0:y1, x0:x1]
                proc.stdin.write(np.ascontiguousarray(out).tobytes())
            done += len(pending)
            if done % (batch_size * 10) < batch_size:
                print(f"[{tag}] frame {done}/{n_frames}")
            pending = []

    proc.stdin.close()
    proc.wait()
    cap.release()
    print(f"[{tag}] wrote {out_path}")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- torch.cuda.is_available() is False")

    args = sys.argv[1:]
    src_dir, pattern, out_dir = args[0], args[1], args[2]
    k1, k2, angle = float(args[3]), float(args[4]), float(args[5])
    ct, cr, cb, cl = float(args[6]), float(args[7]), float(args[8]), float(args[9])

    start_sec, duration_sec = 0.0, None
    if "--demo" in args:
        i = args.index("--demo")
        start_sec, duration_sec = float(args[i + 1]), float(args[i + 2])

    shadow, brightness, blacks = 0.0, 0.0, 0.0
    if "--shadow" in args:
        shadow = float(args[args.index("--shadow") + 1])
    if "--brightness" in args:
        brightness = float(args[args.index("--brightness") + 1])
    if "--blacks" in args:
        blacks = float(args[args.index("--blacks") + 1])

    batch_size = 16
    if "--batch" in args:
        batch_size = int(args[args.index("--batch") + 1])

    parallel = 1
    if "--parallel" in args:
        parallel = int(args[args.index("--parallel") + 1])

    os.makedirs(out_dir, exist_ok=True)
    clips = sorted(glob.glob(os.path.join(src_dir, pattern)))
    if not clips:
        raise SystemExit(f"no clips matched {pattern!r} in {src_dir}")

    def run(src):
        out_path = os.path.join(out_dir, os.path.basename(src))
        if duration_sec is not None:
            base, ext = os.path.splitext(out_path)
            out_path = f"{base}_demo{ext}"
        elif os.path.exists(out_path):
            print(f"[{os.path.basename(src)}] already done, skipping")
            return
        process_one(src, out_path, k1, k2, angle, ct, cr, cb, cl,
                    shadow, brightness, blacks, start_sec, duration_sec, batch_size)

    if parallel <= 1:
        for src in clips:
            run(src)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            list(pool.map(run, clips))


if __name__ == "__main__":
    main()
