r"""apply_lens_correction.py -- batch-apply one camera's locked-in lens correction (barrel
distortion k1/k2, roll rotation, 4-sided crop) to every clip for that camera, frame by frame,
using the EXACT SAME remap/rotate/crop math as lens_calibrator.py's live preview -- so a
production render matches what was actually approved in the GUI, not a re-derivation via
ffmpeg's own lenscorrection filter (a different distortion model, not guaranteed to match
pixel-for-pixel -- see lens_calibrator.py's own docstring for why this matters).

Why per-camera, not per-clip: a GoPro mounted once for the whole show has one fixed lens/roll
correction that applies to every one of its clips -- find it once with lens_calibrator.py on a
single representative frame, then run this script once per camera, not once per clip.

Usage:
  # batch: every clip matching <glob> in <src_dir>, full length
  py apply_lens_correction.py <src_dir> <glob> <out_dir> k1 k2 angle_deg crop_top crop_right crop_bottom crop_left

  # quick demo: one clip, a short snippet only (for checking a calibration before committing
  # to a full-length batch render)
  py apply_lens_correction.py <src_dir> <glob> <out_dir> k1 k2 angle_deg crop_top crop_right crop_bottom crop_left --demo <start_sec> <duration_sec>

  # optional tone adjustments (see lens_calibrator.py's apply_tone) -- the GUI's "Save" prints
  # the exact command line to use, including these flags, once values have been dialed in:
  py apply_lens_correction.py ... --shadow <0.0-1.0> --brightness <-1.0-1.0> --blacks <-1.0-1.0>
"""
import glob
import os
import subprocess
import sys

import cv2

from lens_calibrator import apply_tone, apply_transform, crop_box


def process_one(src, out_path, k1, k2, angle, ct, cr, cb, cl,
                 shadow=0.0, brightness=0.0, blacks=0.0,
                 start_sec=0.0, duration_sec=None):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_ratio = w / h
    x0, y0, x1, y1 = crop_box(w, h, ct, cr, cb, cl, target_ratio)
    out_w, out_h = x1 - x0, y1 - y0
    out_w -= out_w % 2  # libx264 requires even dimensions
    out_h -= out_h % 2
    x1, y1 = x0 + out_w, y0 + out_h

    total_dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0
    dur = duration_sec if duration_sec is not None else max(0.0, total_dur - start_sec)
    n_frames = int(round(dur * fps))
    print(f"{os.path.basename(src)}: {w}x{h} @ {fps:.2f}fps -> {out_w}x{out_h}, "
          f"{start_sec:.1f}s + {dur:.1f}s ({n_frames} frames)")

    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.5f}", "-i", "-",
        "-ss", str(start_sec), "-t", str(dur), "-i", src,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k", "-shortest", out_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            print(f"  stopped early at frame {i}/{n_frames} (end of file)")
            break
        frame = apply_tone(frame, shadow, brightness, blacks)
        out = apply_transform(frame, k1, k2, angle)
        out = out[y0:y1, x0:x1]
        proc.stdin.write(out.tobytes())
        if i % 150 == 0:
            print(f"  frame {i}/{n_frames}")

    proc.stdin.close()
    proc.wait()
    cap.release()
    print(f"  wrote {out_path}")


def main():
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

    os.makedirs(out_dir, exist_ok=True)
    clips = sorted(glob.glob(os.path.join(src_dir, pattern)))
    if not clips:
        raise SystemExit(f"no clips matched {pattern!r} in {src_dir}")

    for src in clips:
        out_path = os.path.join(out_dir, os.path.basename(src))
        if duration_sec is not None:
            base, ext = os.path.splitext(out_path)
            out_path = f"{base}_demo{ext}"
        process_one(src, out_path, k1, k2, angle, ct, cr, cb, cl,
                    shadow, brightness, blacks, start_sec, duration_sec)


if __name__ == "__main__":
    main()
