r"""extract_people.py -- Stage 1 of ai-director: generic (role-agnostic) person+pose detection
and WITHIN-CLIP tracking for every clip in a Masters folder.

stage-movement-analysis's extract_pose.py assumes a fixed Nikon wide shot with exactly 4 band
roles at known stage positions (demo_labels.ROLES) and matches detections to those roles by
stage-position priors. A phone/GoPro clip in a multicam Masters folder has no such fixed
framing -- one person close-up, several people, a pan across the room, a performer who walks
into frame mid-clip. This script makes NO assumption about who or how many people are in frame:
it runs the same YOLO pose model (yolo11x-pose, 10fps, 1080p -- same convention as
stage-movement-analysis, so later signals are comparable) and assigns a stable track_id via
simple nearest-centroid matching across consecutive sampled frames, bridging a short occlusion
gap. This is PER-CLIP, LOCAL tracking only -- it never claims two tracks (even in different
clips from the same camera, let alone different cameras) are the same real person. That
cross-camera/cross-clip identity and performer-attribution problem is Stage 2's job, not this
script's -- see the ai-director SKILL.md and the project plan.

Output per clip, in <out_dir>:
  <clip-basename>.npz
    fps          sampling fps used
    track_id     (D,) int -- one row per detection across all sampled frames
    frame_idx    (D,) int
    box          (D,4) xyxy at 1080p scale
    kp           (D,17,3) COCO keypoints [x,y,conf] at 1080p scale
  <clip-basename>-activity.json
    a per-second movement-energy curve for the clip -- see ACTIVITY NOTES below.

ACTIVITY NOTES: a track's per-frame speed is normalized by its own median shoulder width
(COCO keypoints 5,6) where confidently seen, else its box diagonal (a close-up may never get a
clean two-shoulder view) -- the same "torso widths per second" idea analyze.py uses, just
estimated per-track instead of assumed from a fixed role. Per second, across however many
tracks are actually present, both the MEAN (overall engagement) and MAX (is anyone really
moving right now) are kept raw, not collapsed into one number -- Stage 2's director logic
decides which is more useful per moment, and recomputing from this JSON is cheap.

Usage:
  py extract_people.py <masters_dir> <out_dir> [--fps 10] [--imgsz 1280] [--conf 0.3]
                                                 [--max-gap-frames 15] [--match-dist-frac 0.12]
"""
import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np
from ultralytics import YOLO

from clip_utils import find_clips

W, H = 1920, 1080
CONF_KP = 0.4

def decode_frames(video_path, fps):
    """Yield successive BGR frames at `fps`, scaled to 1080p, via NVDEC (same pattern as
    stage-movement-analysis's extract_pose.py)."""
    dec = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-v", "error", "-hwaccel", "cuda", "-i", video_path,
         "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    fsz = W * H * 3
    while True:
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    dec.wait()


def detections_from_result(res):
    """One YOLO pose result -> list of (centroid(x,y), box(4,), kp(17,3))."""
    out = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    boxes = res.boxes.xyxy.cpu().numpy()
    kp_xy = res.keypoints.xy.cpu().numpy()
    kp_conf = res.keypoints.conf.cpu().numpy()
    for b, kxy, kc in zip(boxes, kp_xy, kp_conf):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        kp = np.concatenate([kxy, kc[:, None]], axis=1)
        out.append((np.array([cx, cy]), b, kp))
    return out


def track_clip(video_path, model, fps, imgsz, conf, max_gap_frames, match_dist):
    """Greedy nearest-centroid tracking across consecutive sampled frames. Returns
    (track_id[D], frame_idx[D], box[D,4], kp[D,17,3])."""
    active = {}       # track_id -> {"centroid": arr, "last_frame": int}
    next_id = 0
    out_tid, out_fidx, out_box, out_kp = [], [], [], []

    frame_idx = 0
    batch_frames = []
    BATCH = 8

    def process_batch(frames, start_idx):
        nonlocal next_id
        results = model(frames, imgsz=imgsz, conf=conf, verbose=False, device=0)
        for off, res in enumerate(results):
            fi = start_idx + off
            dets = detections_from_result(res)
            unmatched_tracks = set(active.keys())
            unmatched_dets = set(range(len(dets)))
            pairs = []
            for tid in unmatched_tracks:
                for di in unmatched_dets:
                    d = float(np.linalg.norm(active[tid]["centroid"] - dets[di][0]))
                    if d <= match_dist:
                        pairs.append((d, tid, di))
            pairs.sort()
            assigned_t, assigned_d = set(), set()
            for d, tid, di in pairs:
                if tid in assigned_t or di in assigned_d:
                    continue
                assigned_t.add(tid)
                assigned_d.add(di)
                active[tid]["centroid"] = dets[di][0]
                active[tid]["last_frame"] = fi
                centroid, box, kp = dets[di]
                out_tid.append(tid); out_fidx.append(fi); out_box.append(box); out_kp.append(kp)
            for di in range(len(dets)):
                if di in assigned_d:
                    continue
                tid = next_id
                next_id += 1
                centroid, box, kp = dets[di]
                active[tid] = {"centroid": centroid, "last_frame": fi}
                out_tid.append(tid); out_fidx.append(fi); out_box.append(box); out_kp.append(kp)
            # drop tracks that have been missing too long
            for tid in list(active.keys()):
                if fi - active[tid]["last_frame"] > max_gap_frames:
                    del active[tid]

    for frame in decode_frames(video_path, fps):
        batch_frames.append(frame)
        if len(batch_frames) == BATCH:
            process_batch(batch_frames, frame_idx - len(batch_frames) + 1)
            frame_idx += 1
            batch_frames = []
        else:
            frame_idx += 1
    if batch_frames:
        process_batch(batch_frames, frame_idx - len(batch_frames))

    if not out_tid:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 4), np.float32),
                np.zeros((0, 17, 3), np.float32))
    return (np.array(out_tid), np.array(out_fidx), np.stack(out_box).astype(np.float32),
            np.stack(out_kp).astype(np.float32))


def shoulder_width(kp_row):
    l, r = kp_row[5], kp_row[6]
    if l[2] > CONF_KP and r[2] > CONF_KP:
        return float(np.hypot(l[0] - r[0], l[1] - r[1]))
    return None


def build_activity(track_id, frame_idx, box, kp, fps):
    """Per-track speed (own-shoulder-widths/sec, falling back to own-box-diagonal/sec),
    then per-second MEAN and MAX across whichever tracks are present that second."""
    if len(track_id) == 0:
        return {"fps": fps, "n_seconds": 0, "mean": [], "max": [], "n_tracks_present": []}
    n_frames = int(frame_idx.max()) + 1
    n_seconds = int(np.ceil(n_frames / fps))
    per_track_speed = {}  # tid -> array[n_frames] of speed (nan where absent)
    tids = sorted(set(track_id.tolist()))
    for tid in tids:
        mask = track_id == tid
        fidx = frame_idx[mask]
        bx = box[mask]
        kpt = kp[mask]
        shoulder_widths = [w for w in (shoulder_width(kpt[i]) for i in range(len(fidx))) if w is not None]
        if shoulder_widths:
            norm = float(np.median(shoulder_widths))
        else:
            diags = [float(np.hypot(bx[i][2] - bx[i][0], bx[i][3] - bx[i][1])) for i in range(len(fidx))]
            norm = float(np.median(diags)) if diags else 1.0
        norm = norm if norm > 1e-3 else 1.0
        speed = np.full(n_frames, np.nan, np.float32)
        order = np.argsort(fidx)
        fidx_s = fidx[order]
        bx_s = bx[order]
        for k in range(1, len(fidx_s)):
            df = fidx_s[k] - fidx_s[k - 1]
            if df <= 0 or df > 5:   # gap too large to call it continuous motion
                continue
            c0 = np.array([(bx_s[k - 1][0] + bx_s[k - 1][2]) / 2, (bx_s[k - 1][1] + bx_s[k - 1][3]) / 2])
            c1 = np.array([(bx_s[k][0] + bx_s[k][2]) / 2, (bx_s[k][1] + bx_s[k][3]) / 2])
            v = float(np.linalg.norm(c1 - c0)) / norm * fps / df
            speed[fidx_s[k]] = v
        per_track_speed[tid] = speed

    mean_curve, max_curve, n_present = [], [], []
    for s in range(n_seconds):
        f0, f1 = int(s * fps), int(min((s + 1) * fps, n_frames))
        vals = []
        for tid in tids:
            seg = per_track_speed[tid][f0:f1]
            seg = seg[~np.isnan(seg)]
            if len(seg):
                vals.append(float(seg.mean()))
        mean_curve.append(float(np.mean(vals)) if vals else None)
        max_curve.append(float(np.max(vals)) if vals else None)
        n_present.append(len(vals))
    return {"fps": fps, "n_seconds": n_seconds, "mean": mean_curve, "max": max_curve,
            "n_tracks_present": n_present}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("masters_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-gap-frames", type=int, default=15)
    ap.add_argument("--match-dist-frac", type=float, default=0.12,
                     help="max centroid jump between sampled frames, as a fraction of frame diagonal")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    match_dist = args.match_dist_frac * float(np.hypot(W, H))
    model = YOLO("yolo11x-pose.pt")
    model.to("cuda")

    clips = find_clips(args.masters_dir)
    print(f"{len(clips)} clips in {args.masters_dir}")
    t_all = time.time()
    for n, path in enumerate(clips, 1):
        base = os.path.splitext(os.path.basename(path))[0]
        out_npz = os.path.join(args.out_dir, base + ".npz")
        out_act = os.path.join(args.out_dir, base + "-activity.json")
        if os.path.exists(out_npz) and os.path.exists(out_act):
            print(f"[{n}/{len(clips)}] skip {base}", flush=True)
            continue
        t0 = time.time()
        tid, fidx, box, kp = track_clip(path, model, args.fps, args.imgsz, args.conf,
                                         args.max_gap_frames, match_dist)
        np.savez_compressed(out_npz, fps=args.fps, track_id=tid, frame_idx=fidx, box=box, kp=kp)
        activity = build_activity(tid, fidx, box, kp, args.fps)
        with open(out_act, "w") as fh:
            json.dump(activity, fh)
        n_tracks = len(set(tid.tolist())) if len(tid) else 0
        print(f"[{n}/{len(clips)}] {base}: {len(tid)} detections, {n_tracks} tracks, "
              f"{activity['n_seconds']}s ({time.time() - t0:.0f}s)", flush=True)
    print(f"ALL DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
