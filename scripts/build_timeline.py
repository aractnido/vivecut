r"""build_timeline.py -- Stage 1 of ai-director: the final deliverable tying every camera's
clips and per-clip signals together on ONE real-world UTC timeline.

Every other Stage-1 script works in a camera's own local/internal time (a clip's own
creation_time, a clip's own frame_idx). This script is where that stops being useful and
everything needs to live on the same axis: "what was happening, for real, at 19:42:10 UTC,
across every camera that was rolling" is the question Stage 2's director logic will actually
ask.

The one camera that does NOT have its own real-world clock baked into its files is the Nikon
continuous master: concert-video-pipeline's Songs-4K output is split and numbered by a human
watching the INTERNAL recording clock (Canciones.txt, "01-Summer Of 69,00:53:13" = 53:13 into
the raw recording), with no creation_time tag anywhere. Real-world time for the Nikon is
recovered by an AFFINE FIT against the one camera whose clock is trusted (the reference
--song-windows CSV, normally GoPro2 here): for every song number present in BOTH Canciones.txt
and the song-windows CSV, (real_utc_start - internal_seconds) should be the same constant if the
Nikon's internal clock runs at a true, non-drifting rate -- confirmed for Give 2026 across songs
01/13/22 (start, middle, end of the show) to within a few seconds, well inside the slop of a
human reading a minute:second timestamp off a scrubber. The median of all matched songs' offsets
is used as the one constant, not just a single anchor, so an occasional few-second misread
doesn't skew the whole show.

This script does NOT embed every signal inline -- some sidecars (a zoom/glow trace, a depth
npz) are large and binary. The timeline is an INDEX: for every clip, on every camera, its real
[start, end) window plus the file paths of whichever per-clip signal files happen to exist for
it yet (this is meant to be re-run as more Stage-1 scripts finish; nothing here requires every
signal to be present first).

Usage:
  py build_timeline.py --out timeline.json \
    --masters-manifest <Masters>/manifest.json \
    --song-windows <song_windows.csv> \
    --canciones <Canciones.txt> \
    --nikon-songs <Songs-4K dir> \
    --metadata-dir <Metadata>        (bootlegcam sidecars: -audioquality/-blackout/-zoomglow)
    --audioenergy-dir <AudioEnergy>  (this skill's audio_energy.py output)
    --people-dir <People>            (this skill's extract_people.py output)
    --depth-dir <Depth>              (this skill's extract_depth.py output, optional)
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
from datetime import datetime, timedelta


def ffprobe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    return float(r.stdout.strip())


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_song_windows(csv_path):
    songs = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row:
                continue
            name, ct, dur = row[0], row[1], row[2]
            num = name.split("-")[0]
            start = parse_iso(ct)
            end = start + timedelta(seconds=float(dur))
            songs[num] = {"start": start, "end": end}
    return songs


CANCIONES_RE = re.compile(r'^\s*(.+?),(\d{2}):(\d{2}):(\d{2})\s*$')


def load_canciones(path):
    """Returns [{num, title, is_skip, internal_start_s}], sorted by internal_start_s. `num` is
    None for a SKIP- line."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = CANCIONES_RE.match(line)
            if not m:
                continue
            label, hh, mm, ss = m.groups()
            secs = int(hh) * 3600 + int(mm) * 60 + int(ss)
            is_skip = label.startswith("SKIP-")
            num = None
            title = label
            if not is_skip:
                mm2 = re.match(r'^(\d+)-(.+)$', label)
                if mm2:
                    num, title = mm2.group(1), mm2.group(2)
            entries.append({"num": num, "title": title, "is_skip": is_skip, "internal_start_s": secs})
    return sorted(entries, key=lambda e: e["internal_start_s"])


def fit_nikon_offset(canciones, song_windows):
    """Median of (real_utc_start - internal_start_s) over every song number present in both
    Canciones.txt and the reference song-windows CSV. Returns (offset_seconds, matches) where
    matches is the per-song list used, for transparency/verification."""
    matches = []
    for e in canciones:
        if e["is_skip"] or e["num"] not in song_windows:
            continue
        real_start = song_windows[e["num"]]["start"]
        epoch_s = (real_start - datetime(1970, 1, 1, tzinfo=real_start.tzinfo)).total_seconds()
        offset = epoch_s - e["internal_start_s"]
        matches.append({"num": e["num"], "title": e["title"], "offset_s": offset})
    offsets = sorted(m["offset_s"] for m in matches)
    if not offsets:
        raise RuntimeError("No song numbers matched between Canciones.txt and --song-windows; "
                            "can't fit the Nikon affine offset.")
    mid = len(offsets) // 2
    median_offset = offsets[mid] if len(offsets) % 2 else (offsets[mid - 1] + offsets[mid]) / 2
    spread = offsets[-1] - offsets[0]
    return median_offset, matches, spread


def epoch_to_iso(epoch_s):
    return (datetime(1970, 1, 1) + timedelta(seconds=epoch_s)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sidecar_if_exists(d, base, suffix):
    if not d:
        return None
    p = os.path.join(d, base + suffix)
    return p if os.path.exists(p) else None


def build_nikon_clips(canciones, nikon_songs_dir, offset_s, metadata_dir, audioenergy_dir,
                       people_dir, depth_dir):
    clips = []
    # Carry each entry's own position in the FULL list alongside it, rather than recovering it
    # later via canciones.index(e) -- that matched by value equality, so two entries with the
    # same label and internal_start_s (an accidental duplicate line) resolved to the same index
    # for both, silently corrupting internal_end_s for one of them.
    non_skip = [(idx, e) for idx, e in enumerate(canciones) if not e["is_skip"]]
    for idx_full, e in non_skip:
        num, title = e["num"], e["title"]
        # internal end = start of the NEXT entry in the full (skip-inclusive) list, if any
        internal_end_s = canciones[idx_full + 1]["internal_start_s"] if idx_full + 1 < len(canciones) else None
        real_start_epoch = e["internal_start_s"] + offset_s
        real_end_epoch = internal_end_s + offset_s if internal_end_s is not None else None
        clip = {
            "song_num": num, "title": title,
            "real_start_utc": epoch_to_iso(real_start_epoch),
            "real_end_utc": epoch_to_iso(real_end_epoch) if real_end_epoch else None,
        }
        if nikon_songs_dir:
            matches = glob.glob(os.path.join(nikon_songs_dir, f"{num}-*.mp4"))
            if matches:
                clip["file"] = matches[0]
                base = os.path.splitext(os.path.basename(matches[0]))[0]
                clip["audioquality"] = sidecar_if_exists(metadata_dir, base, "-audioquality.json")
                clip["blackout"] = sidecar_if_exists(metadata_dir, base, "-blackout.json")
                clip["zoomglow"] = sidecar_if_exists(metadata_dir, base, "-zoomglow.json")
                clip["audioenergy"] = sidecar_if_exists(audioenergy_dir, base, "-audioenergy.json")
                clip["people"] = sidecar_if_exists(people_dir, base, ".npz")
                clip["activity"] = sidecar_if_exists(people_dir, base, "-activity.json")
                clip["depth"] = sidecar_if_exists(depth_dir, base, "-depth.npz")
        clips.append(clip)
    return clips


def build_other_camera_clips(manifest, metadata_dir, audioenergy_dir, people_dir, depth_dir):
    by_camera = {}
    for c in manifest["clips"]:
        base = os.path.splitext(c["file"])[0]
        entry = dict(c)
        entry["audioquality"] = sidecar_if_exists(metadata_dir, base, "-audioquality.json")
        entry["blackout"] = sidecar_if_exists(metadata_dir, base, "-blackout.json")
        entry["zoomglow"] = sidecar_if_exists(metadata_dir, base, "-zoomglow.json")
        entry["audioenergy"] = sidecar_if_exists(audioenergy_dir, base, "-audioenergy.json")
        entry["people"] = sidecar_if_exists(people_dir, base, ".npz")
        entry["activity"] = sidecar_if_exists(people_dir, base, "-activity.json")
        entry["depth"] = sidecar_if_exists(depth_dir, base, "-depth.npz")
        by_camera.setdefault(c["source"], []).append(entry)
    for cam in by_camera:
        by_camera[cam].sort(key=lambda e: e["start"])
    return by_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--masters-manifest", required=True)
    ap.add_argument("--song-windows", required=True)
    ap.add_argument("--canciones", required=True)
    ap.add_argument("--nikon-songs", default=None)
    ap.add_argument("--metadata-dir", default=None)
    ap.add_argument("--audioenergy-dir", default=None)
    ap.add_argument("--people-dir", default=None)
    ap.add_argument("--depth-dir", default=None)
    args = ap.parse_args()

    with open(args.masters_manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    song_windows = load_song_windows(args.song_windows)
    canciones = load_canciones(args.canciones)
    offset_s, matches, spread = fit_nikon_offset(canciones, song_windows)
    print(f"Nikon affine offset: {offset_s:.1f}s (median over {len(matches)} matched songs, "
          f"spread {spread:.1f}s)")
    for m in matches:
        print(f"  song {m['num']:>3} {m['title']:35s} offset {m['offset_s']:.1f}s")

    nikon_clips = build_nikon_clips(canciones, args.nikon_songs, offset_s, args.metadata_dir,
                                     args.audioenergy_dir, args.people_dir, args.depth_dir)
    other = build_other_camera_clips(manifest, args.metadata_dir, args.audioenergy_dir,
                                      args.people_dir, args.depth_dir)

    timeline = {
        "songs": manifest["songs"],
        "nikon_affine": {"offset_seconds": offset_s, "spread_seconds": spread,
                          "matched_songs": matches,
                          "note": "real_utc_epoch = nikon_internal_seconds + offset_seconds"},
        "cameras": {"nikon": nikon_clips, **other},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)
    total_clips = len(nikon_clips) + sum(len(v) for v in other.values())
    print(f"\nWrote timeline with {total_clips} clips across {1 + len(other)} cameras -> {args.out}")


if __name__ == "__main__":
    main()
