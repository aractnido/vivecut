r"""apply_stabilization.py -- stabilize ONLY the clips a shakiness-report.csv (see
detect_shakiness.py) flags as actually shaky, via ffmpeg's two-pass libvidstab
(vidstabdetect + vidstabtransform). Clips at/below the threshold are passed through
untouched (hardlinked, no re-encode) -- stabilization has a real visual cost (slight crop,
occasional warping on fast content) that isn't worth paying on footage that's already steady.

Usage:
  py apply_stabilization.py <masters_dir> <shakiness_csv> <out_dir> [--threshold 12.0] [--parallel 3]
"""
import csv
import glob
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

from clip_utils import find_clips


def esc_path(p):
    # ffmpeg filter option values use ':' as a key=value separator, so a Windows drive
    # letter's literal ':' has to be escaped, and backslashes read more reliably as forward
    # slashes in filter strings -- same two-step fix used throughout this project.
    return p.replace("\\", "/").replace(":", "\\:")


def quote_val(p):
    # Wrapping in '...' protects spaces. A literal "'" in the value is a dead end, though --
    # tried the documented close-quote/escaped-quote/reopen-quote trick (' -> '\'') and ffmpeg's
    # parser did NOT treat it as one literal quote character; it swallowed the rest of the
    # filtergraph string as part of this value instead (confirmed: vidstabtransform tried to
    # open a "file" literally named "...mp4.trf:zoom=0:smoothing=15"). So this function assumes
    # its input has no apostrophe -- see tmp_trf_path() below, which guarantees that by never
    # building the .trf sidecar path from the clip's own (real-world, apostrophe-prone) name.
    assert "'" not in p, f"quote_val can't safely escape an embedded quote: {p!r}"
    return "'" + esc_path(p) + "'"


def tmp_trf_path(out_dir):
    # Deliberately NOT derived from the clip's own filename: that name can contain spaces,
    # apostrophes, accented characters -- anything a real-world clip title throws at it -- and
    # this path has to survive being embedded inside an ffmpeg filtergraph string (quote_val),
    # which only spaces survive safely. src/out_path never go through that string (they're
    # plain argv to -i and the output path), so only THIS one sidecar path needs to be simple.
    return os.path.join(out_dir, f"_stab_{uuid.uuid4().hex}.trf")


def stabilize(src, out_path, tmp_trf, zoom=0, smoothing=15):
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                         "-vf", f"vidstabdetect=shakiness=8:accuracy=15:result={quote_val(tmp_trf)}",
                         "-f", "null", "-"])
    if r.returncode != 0:
        return False
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                         "-vf", f"vidstabtransform=input={quote_val(tmp_trf)}:zoom={zoom}:smoothing={smoothing},unsharp=5:5:0.8:3:3:0.4",
                         "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "18",
                         "-b:v", "0", "-pix_fmt", "yuv420p",
                         "-c:a", "copy", out_path])
    if os.path.exists(tmp_trf):
        os.remove(tmp_trf)
    return r.returncode == 0


def main():
    masters_dir, csv_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    threshold = 12.0
    if "--threshold" in sys.argv:
        threshold = float(sys.argv[sys.argv.index("--threshold") + 1])
    # zoom/smoothing defaults match what this project shipped with originally (0/15); exposed as
    # flags once a real comparison (see SKILL.md) showed those weren't aggressive enough for
    # Nova's shakier clips -- e.g. --smoothing 30 --zoom 5, picked by watching a multi-level
    # side-by-side, not guessed.
    zoom = 0
    if "--zoom" in sys.argv:
        zoom = float(sys.argv[sys.argv.index("--zoom") + 1])
    smoothing = 15
    if "--smoothing" in sys.argv:
        smoothing = float(sys.argv[sys.argv.index("--smoothing") + 1])
    # --only restricts to specific basenames (comma-separated) -- for redoing/testing one or two
    # clips with new settings without reprocessing a whole camera's batch (and without risking a
    # collision with some OTHER process concurrently reading the rest of out_dir, e.g. a render
    # in progress -- hit that exact race condition once already in this project, see SKILL.md).
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    # Clips processed concurrently -- the actual ffmpeg work (vidstab's CPU-bound motion
    # estimation, then NVENC encode) releases the GIL during subprocess.run(), so this is real
    # overlap, not just interleaving, same pattern as apply_lens_correction_gpu.py's --parallel.
    parallel = 1
    if "--parallel" in sys.argv:
        parallel = int(sys.argv[sys.argv.index("--parallel") + 1])

    shakiness = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shakiness[row["file"]] = float(row["jitter_std_px"])

    os.makedirs(out_dir, exist_ok=True)
    clips = find_clips(masters_dir)
    if only:
        clips = [c for c in clips if os.path.basename(c) in only]
    def run(i, src):
        base = os.path.basename(src)
        out_path = os.path.join(out_dir, base)
        jitter = shakiness.get(base)
        if jitter is not None and jitter > threshold:
            print(f"[{i+1}/{len(clips)}] STABILIZING {base} (jitter={jitter:.1f}px, zoom={zoom} smoothing={smoothing})")
            tmp_trf = tmp_trf_path(out_dir)
            ok = stabilize(src, out_path, tmp_trf, zoom=zoom, smoothing=smoothing)
            print(f"  {'OK' if ok else 'FAILED'}")
            return "stab" if ok else None
        detail = f"(jitter={jitter:.1f}px)" if jitter is not None else "(no score)"
        print(f"[{i+1}/{len(clips)}] pass-through {base} {detail}")
        if not os.path.exists(out_path):
            os.link(src, out_path)
        return "pass"

    if parallel <= 1:
        results = [run(i, src) for i, src in enumerate(clips)]
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            results = list(pool.map(lambda a: run(*a), enumerate(clips)))

    n_stab = sum(1 for r in results if r == "stab")
    n_pass = sum(1 for r in results if r == "pass")
    print(f"done: {n_stab} stabilized, {n_pass} passed through unchanged")


if __name__ == "__main__":
    main()
