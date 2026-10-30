r"""sync_by_song_window.py -- measure a clip's true real-world start by GCC-PHAT audio correlation
against the FULL SONG window (from the clip's own song-number filename prefix), not against a
margin around the clip's existing metadata.

Why this exists: precise_audio_sync.py's search window is a small margin (default 20s) around the
clip's EXISTING metadata offset -- correct when that metadata is roughly right and only needs a
small touch-up (worked for Iris, GoPro1/2). It's useless for a camera whose metadata is wrong by
much more than the margin (Nova's Android/OneDrive clips: 56-90+s off, see SKILL.md) -- a
narrow search around a badly-wrong anchor just can't reach the true answer.

The fix: every clip's filename already encodes which song it belongs to (the "NN-" prefix, same
Masters convention every camera uses). That song's real-world window is independently known and
validated (Nikon's own per-song split, cross-checked across all 22 songs, see
build_timeline.py/SKILL.md) -- a MUCH more reliable anchor than this one camera's own bad clock.
So: search the clip's audio against the ENTIRE song's Nikon audio (not a +-20s sliver), using that
song's real start as the reference point. This is still a bounded search (one song's length, a few
minutes) -- nowhere near the unconstrained whole-show search that caused the original aliasing bug.

Still get a visual sanity check, not just audio: a repeated chorus WITHIN one song can still fool
GCC-PHAT (same failure class as the original whole-show aliasing, just lower-probability at this
scale). After measuring, this script extracts one Nikon frame and one clip frame at the computed
offset and saves them side by side so a human (or a quick second pass) can eyeball whether the two
cameras are looking at the same moment -- same discipline as the karaoke-lyric frame method already
used elsewhere in this project, just automated into the normal workflow instead of a one-off.

Usage:
  py sync_by_song_window.py <timeline.json> <camera> <masters_dir> [--apply] [--min-z 8.0] [--only file.mp4,...]

Without --apply: reports each clip's current metadata offset, the measured true offset, delta, and
z-score, plus writes a frame-pair PNG per clip for visual confirmation. With --apply: writes the
correction into timeline.json for every clip whose z-score clears --min-z.
"""
import glob
import json
import os
import subprocess
import sys
from datetime import timedelta

import numpy as np

from audio_sync_utils import SR, aud, fmt, gcc_phat, parse

FRAME_OUT_DIR = os.path.join(
    os.environ.get("VIVECUT_SHOW_ROOT", r"C:\path\to\your\show-working-folder"),
    "Preview", "sync_frames")


def song_num_from_filename(fname):
    # Masters convention: "NN-Title[-X].ext" or a bridge "NN-MM-Title to Title-X.ext" -- only the
    # FIRST song number is needed here (a bridge clip's own content starts in that first song).
    return fname.split("-", 1)[0]


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                        capture_output=True, text=True)
    return float(r.stdout.strip())


def save_frame_pair(nikon_file, nikon_t, clip_file, clip_t, out_png):
    os.makedirs(FRAME_OUT_DIR, exist_ok=True)
    a = os.path.join(FRAME_OUT_DIR, "_a.png")
    b = os.path.join(FRAME_OUT_DIR, "_b.png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(max(0, nikon_t)), "-i", nikon_file,
                     "-frames:v", "1", a])
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(max(0, clip_t)), "-i", clip_file,
                     "-frames:v", "1", b])
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", a, "-i", b, "-filter_complex",
                     "[0:v]scale=-1:720[l];[1:v]scale=-1:720[r];[l][r]hstack=2", out_png])


def main():
    args = sys.argv[1:]
    timeline_path, camera, masters_dir = args[0], args[1], args[2]
    apply = "--apply" in args
    min_z = 8.0
    if "--min-z" in args:
        min_z = float(args[args.index("--min-z") + 1])
    only = None
    if "--only" in args:
        only = set(args[args.index("--only") + 1].split(","))

    tl = json.load(open(timeline_path, encoding="utf-8"))
    nikon_by_song = {c["song_num"]: c for c in tl["cameras"]["nikon"]}

    results = []
    for c in tl["cameras"][camera]:
        if only and c["file"] not in only:
            continue
        song_num = song_num_from_filename(c["file"])
        nc = nikon_by_song.get(song_num)
        if not nc:
            print(f"SKIP {c['file']}: no nikon song {song_num}")
            continue

        path = os.path.join(masters_dir, c["file"])
        if not os.path.exists(path):
            matches = glob.glob(os.path.join(masters_dir, c["file"]))
            if not matches:
                print(f"SKIP (file not found): {c['file']}")
                continue
            path = matches[0]

        meta_start = parse(c["start"])
        nikon_start = parse(nc["real_start_utc"])
        meta_offset = (meta_start - nikon_start).total_seconds()

        clip_dur = probe_duration(path)
        song_dur = probe_duration(nc["file"])

        sig = aud(path, 0, min(clip_dur, 40))
        ref = aud(nc["file"], 0, song_dur)
        lag, z = gcc_phat(sig, ref)
        # gcc_phat(sig, ref) returns the lag at which ref's content (ref[0] = nikon_local 0) is
        # found WITHIN sig (the clip, from its own local 0) -- i.e. clip_local(lag) == nikon_local(0),
        # so the clip's OWN t=0 maps to nikon_local(-lag), not +lag. Got this backwards on the first
        # pass (caught by cross-checking against the already frame-verified 01-Summer-of-69-C offset:
        # this sign gave -2.70s, the correct one gives +2.70s, matching the frame-verified +2.25s).
        measured_offset = -lag  # seconds into the song where this clip's audio starts
        true_start = nikon_start + timedelta(seconds=measured_offset)
        delta = (true_start - meta_start).total_seconds()

        results.append((c["file"], song_num, meta_offset, measured_offset, delta, z))
        print(f"{c['file']:45s} song={song_num}  meta_offset={meta_offset:8.2f}  "
              f"measured_offset={measured_offset:8.2f}  delta={delta:+8.2f}  z={z:6.1f}")

        out_png = os.path.join(FRAME_OUT_DIR, f"songwindow_{camera}_{song_num}_{os.path.splitext(c['file'])[0]}.png".replace(" ", "_"))
        save_frame_pair(nc["file"], measured_offset, path, 0.5, out_png)
        print(f"  frame pair -> {out_png}")

    confident = [r for r in results if r[5] >= min_z]
    low_conf = [r for r in results if r[5] < min_z]
    print(f"\n{len(confident)}/{len(results)} clips confident (z>={min_z})")
    if low_conf:
        print(f"low-confidence (left uncorrected if --apply): {[r[0] for r in low_conf]}")

    if apply:
        by_file = {r[0]: r[4] for r in confident}
        n = 0
        for c in tl["cameras"][camera]:
            if c["file"] in by_file:
                delta = by_file[c["file"]]
                c["start"] = fmt(parse(c["start"]) + timedelta(seconds=delta))
                c["end"] = fmt(parse(c["end"]) + timedelta(seconds=delta))
                n += 1
        json.dump(tl, open(timeline_path, "w", encoding="utf-8"), indent=2)
        print(f"\napplied corrections to {n} {camera} clips in {timeline_path}")


if __name__ == "__main__":
    main()
