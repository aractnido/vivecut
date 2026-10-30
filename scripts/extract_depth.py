r"""extract_depth.py -- Stage 1 of ai-director: per-clip relative depth, generalized beyond
stage-movement-analysis's fixed-4-role version (which samples depth only at 4 known band-role
keypoint slots on a single fixed Nikon wide shot).

A multicam Masters clip has no fixed roster/framing -- extract_people.py already produced
generic (unidentified, per-clip-local) person tracks for every clip. This script reuses that
output as WHERE to sample depth, instead of 4 fixed role slots: for every detection recorded by
extract_people.py, it samples the depth map at that detection's head point (falling back to its
box centre when the head/face keypoints weren't confident) and its box centre.

It also writes a per-clip "room depth profile" independent of any detected person: a coarse
downsampled grid of the clip's median depth map. The user's own motivation for this (conference
room, boxy, no stage) is understanding each camera's relative distances well enough to inform
zoom/crop decisions later, not just how far a given performer is -- e.g. "this camera's left
third of frame is all near wall, don't crop there" is a property of the ROOM as shot by that
camera, not of any one performer.

Model: Video Depth Anything (vits checkpoint), vendored under stage-movement-analysis's
assets/ -- reused as-is via its path, not copied, so there's one copy of the ~112MB checkpoint.
Needs that skill's .venv-depth environment (torch cu128 + the vendored repo), NOT the main
ultralytics venv extract_people.py uses. Convention (verified in that skill): HIGHER value =
CLOSER to the camera.

Output per clip, in <out_dir>: <clip-basename>-depth.npz
  frame_idx         (D,) int, aligned 1:1 with extract_people.py's own frame_idx for this clip
  depth_head        (D,) float32, NaN where no confident head point
  depth_box         (D,) float32, sampled at the detection's box centre
  room_grid         (gh,gw) float32, median depth over the whole clip, coarse grid
  room_grid_shape   (gh, gw)

Usage:
  py extract_depth.py <masters_dir> <people_dir> <out_dir> [--max-res 640]
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

from clip_utils import find_clips

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE_SKILL = os.path.join(HERE, "..", "..", "stage-movement-analysis")
VDA_DIR = os.path.join(STAGE_SKILL, "assets", "video-depth-anything")
CHECKPOINT = os.path.join(VDA_DIR, "checkpoints", "video_depth_anything_vits.pth")

sys.path.insert(0, VDA_DIR)
import torch  # noqa: E402
from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402

GRID_W, GRID_H = 12, 7
CONF_KP = 0.4




def read_video_nvdec(video_path, max_w):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,width,height", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True).stdout.strip()
    w_str, h_str, fr_str = probe.rstrip(",").split(",")
    num, den = fr_str.split("/")
    fps = float(num) / float(den)
    src_w, src_h = int(w_str), int(h_str)
    out_w = max_w
    out_h = int(round(src_h * (max_w / src_w) / 2) * 2)

    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-v", "error", "-hwaccel", "cuda", "-i", video_path,
         "-vf", f"scale={out_w}:{out_h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = out_w * out_h * 3
    chunks = []
    while True:
        chunk = proc.stdout.read(4 * 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    proc.wait()
    buf = b"".join(chunks)
    n_frames = len(buf) // frame_bytes
    frames = np.frombuffer(buf, dtype=np.uint8, count=n_frames * frame_bytes).reshape(n_frames, out_h, out_w, 3)
    return frames, fps


def head_point(kp_row):
    pts = kp_row[:5]
    ok = pts[:, 2] > CONF_KP
    if ok.sum() < 2:
        return None
    return float(np.mean(pts[ok, 0])), float(np.mean(pts[ok, 1]))


def process_clip(model, device, video_path, people_npz, out_path, max_res, log):
    t0 = time.time()
    frames, fps = read_video_nvdec(video_path, max_res)
    if frames.shape[0] == 0:
        log(f"  no frames decoded, skipping")
        return
    log(f"  read {frames.shape[0]} frames @ {fps:.2f}fps, {frames.shape[2]}x{frames.shape[1]} "
        f"({time.time() - t0:.0f}s)")
    t1 = time.time()
    with torch.no_grad():
        depths, out_fps = model.infer_video_depth(frames, fps, input_size=518, device=device, fp32=False)
    log(f"  depth inferred: {depths.shape} ({time.time() - t1:.0f}s)")
    DH, DW = depths.shape[1], depths.shape[2]

    d = np.load(people_npz)
    frame_idx_people = d["frame_idx"]
    box = d["box"]
    kp = d["kp"]
    people_fps = float(d["fps"]) if "fps" in d else 10.0
    # extract_people.py samples at 1920x1080 scale -- find the scale factor into this depth frame
    scale_x = DW / 1920.0
    scale_y = DH / 1080.0

    n = len(frame_idx_people)
    depth_head = np.full(n, np.nan, dtype=np.float32)
    depth_box = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        t_sec = frame_idx_people[i] / people_fps
        fidx = int(round(t_sec * out_fps))
        if fidx < 0 or fidx >= depths.shape[0]:
            continue
        hp = head_point(kp[i])
        if hp is not None:
            x, y = hp
            dxp = int(np.clip(x * scale_x, 0, DW - 1))
            dyp = int(np.clip(y * scale_y, 0, DH - 1))
            depth_head[i] = depths[fidx, dyp, dxp]
        bx0, by0, bx1, by1 = box[i]
        if not np.isnan(bx0):
            cxp = int(np.clip((bx0 + bx1) / 2 * scale_x, 0, DW - 1))
            cyp = int(np.clip((by0 + by1) / 2 * scale_y, 0, DH - 1))
            depth_box[i] = depths[fidx, cyp, cxp]

    # room depth profile: median depth over the whole clip, downsampled to a coarse grid
    median_depth = np.median(depths, axis=0)  # (DH, DW)
    gh_edges = np.linspace(0, DH, GRID_H + 1).astype(int)
    gw_edges = np.linspace(0, DW, GRID_W + 1).astype(int)
    room_grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            cell = median_depth[gh_edges[gy]:gh_edges[gy + 1], gw_edges[gx]:gw_edges[gx + 1]]
            room_grid[gy, gx] = float(cell.mean()) if cell.size else np.nan

    np.savez_compressed(out_path, frame_idx=frame_idx_people, depth_head=depth_head,
                         depth_box=depth_box, room_grid=room_grid,
                         room_grid_shape=np.array([GRID_H, GRID_W]), out_fps=out_fps, max_res=max_res)
    del depths, frames
    log(f"  saved {out_path} ({time.time() - t0:.0f}s total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("masters_dir")
    ap.add_argument("people_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--max-res", type=int, default=640)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}, max_res={args.max_res}")
    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    log("model loaded")

    clips = find_clips(args.masters_dir)
    log(f"{len(clips)} clips")
    t_start = time.time()
    done, skipped = 0, 0
    for i, path in enumerate(clips):
        base = os.path.splitext(os.path.basename(path))[0]
        people_npz = os.path.join(args.people_dir, base + ".npz")
        out_path = os.path.join(args.out_dir, base + "-depth.npz")
        if os.path.exists(out_path):
            skipped += 1
            continue
        if not os.path.exists(people_npz):
            log(f"[{i + 1}/{len(clips)}] {base}: no people npz yet, skipping (run extract_people.py first)")
            continue
        log(f"[{i + 1}/{len(clips)}] {base}: starting")
        try:
            process_clip(model, device, path, people_npz, out_path, args.max_res, log)
            done += 1
        except Exception as e:
            log(f"  ERROR on {base}: {e!r}")
    log(f"done: {done} processed, {skipped} already had output, {time.time() - t_start:.0f}s total")


if __name__ == "__main__":
    main()
