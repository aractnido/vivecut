"""
build_masters.py

Stage 1 of the ai-director skill: build a single Masters\ folder (bootlegcam's own
"NN-Title[-X].ext" convention) out of several independent camera/phone sources, using
each clip's REAL recording timestamp (not filename order) to assign it to a song and,
for already-labeled contributors, to sanity-check the label.

Show-agnostic: every source is a CLI argument, nothing about a specific show/venue/band
is hardcoded here. One run = one show.

Song windows (the ground truth every other camera is checked/assigned against) come from
a CSV with columns: song,creation_time,duration -- i.e. exactly what a camera that already
auto-labels per-song files (e.g. a GoPro set to chapter-per-song) produces directly via
ffprobe. See `make_song_windows.py` to build this CSV from such a camera's own folder.

Usage:
  py build_masters.py --out <Masters dir> --song-windows windows.csv ^
    --labeled gopro2=<dir> ^
    --labeled nova=<dir> ^
    --raw iris=<dir> ^
    --offset-corrected "gopro1=<dir>=<offset_seconds>"

  --labeled <name>=<dir>   Clips already named "NN-Title...ext" (any suffix after the
                           title is ignored/normalized away). Real timestamp is read and
                           checked against --song-windows as a sanity check only.
  --raw <name>=<dir>       Clips with no song label (e.g. raw "yyyymmdd_HHMMSS...ext").
                           Assigned to song(s) purely by real-timestamp overlap against
                           --song-windows. A clip overlapping two songs becomes a bridge
                           clip, named "NN-MM-Title-X.ext" per bootlegcam's convention.
  --offset-corrected "<name>=<dir>=<seconds>"
                           Like --raw, but every clip's creation_time is first corrected
                           by adding <seconds> (can be negative) -- for a camera whose
                           clock was set wrong by a constant amount. The corrected
                           timestamp is also re-stamped into the output file's own
                           metadata (lossless remux), never just assumed in memory, so
                           every later stage reads a self-consistent file.

Output: <out>\<NN-Title[-X] or NN-MM-Title-X>.ext, renamed/remuxed as needed, plus
<out>\manifest.json listing every clip's final name, source contributor, and real
[start, end) window -- the clip-level backbone of the Stage 1 unified timeline.
"""
import argparse, json, os, re, shutil, subprocess, sys
from datetime import datetime, timedelta

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABELED_RE = re.compile(r'^(\d+)\s*-\s*(.+?)(?:\s*-\s*\d+)?$')  # "NN-Title", "NN - Title", "NN-Title - 2"


def ffprobe(field, path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", field,
                         "-of", "default=nw=1:nk=1", path],
                        capture_output=True, text=True)
    return r.stdout.strip()


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_song_windows(csv_path):
    import csv as csv_mod
    songs = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        r = csv_mod.reader(f)
        next(r)  # header
        for row in r:
            if not row:
                continue
            name, ct, dur = row[0], row[1], row[2]
            num = name.split("-")[0]
            title = name[len(num) + 1:]
            title = re.sub(r"\.\w+$", "", title)            # strip file extension
            title = re.sub(r"\s*-\s*Gx\d+.*$", "", title, flags=re.IGNORECASE)  # strip a GoPro-style suffix (GX is GoPro's own convention, not lowercase Gx)
            start = parse_iso(ct)
            end = start + timedelta(seconds=float(dur))
            songs.append({"num": num, "title": title.strip(), "start": start, "end": end})
    return sorted(songs, key=lambda s: s["start"])


def sanitize(name):
    invalid = '<>:"/\\|?*'
    for c in invalid:
        name = name.replace(c, " ")
    return re.sub(r"\s+", " ", name).strip()


def remux_with_offset(src, dst, offset_seconds):
    ct = ffprobe("format_tags=creation_time", src)
    corrected = parse_iso(ct) + timedelta(seconds=offset_seconds)
    corrected_str = corrected.strftime("%Y-%m-%dT%H:%M:%S.%fZ")  # %f preserves real sub-second precision
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-i", src, "-map_metadata", "0", "-map", "0", "-c", "copy",
                     "-metadata", f"creation_time={corrected_str}", dst], check=True)
    return corrected


def assign_overlaps(start, end, songs):
    return [s for s in songs if s["start"] < end and s["end"] > start]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--song-windows", required=True)
    ap.add_argument("--labeled", action="append", default=[], metavar="name=dir")
    ap.add_argument("--raw", action="append", default=[], metavar="name=dir")
    ap.add_argument("--offset-corrected", action="append", default=[], metavar="name=dir=seconds")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    songs = load_song_windows(args.song_windows)
    print(f"Loaded {len(songs)} song windows from {args.song_windows}")

    manifest = []
    # Letters are unique PER SONG BUCKET, not per contributor -- a single contributor
    # routinely has several clips for the same song (restarted recording, multi-clip
    # coverage), and bootlegcam's own "-X" suffix must be unique within that song's
    # folder listing, not just unique per source.
    song_letter_idx = {}

    def next_letter(song_key):
        idx = song_letter_idx.get(song_key, 0)
        song_letter_idx[song_key] = idx + 1
        if idx >= len(LETTERS):
            raise RuntimeError(f"More than {len(LETTERS)} clips for song bucket {song_key!r}")
        return LETTERS[idx]

    # --- labeled contributors: copy + normalize suffix, sanity-check timestamp ---
    for spec in args.labeled:
        name, src_dir = spec.split("=", 1)
        for fn in sorted(os.listdir(src_dir)):
            if not fn.lower().endswith((".mp4", ".mov")):
                continue
            m = LABELED_RE.match(os.path.splitext(fn)[0])
            if not m:
                print(f"  SKIP (no NN-Title prefix): {name}/{fn}")
                continue
            num, title = m.group(1), sanitize(m.group(2))
            src = os.path.join(src_dir, fn)
            ct = ffprobe("format_tags=creation_time", src)
            dur = float(ffprobe("format=duration", src))
            start = parse_iso(ct)
            end = start + timedelta(seconds=dur)
            match = next((s for s in songs if s["num"] == num), None)
            if match and not (match["start"] - timedelta(seconds=30) <= start <= match["end"] + timedelta(seconds=30)):
                print(f"  WARN: {name}/{fn} labeled song {num} but real time {ct} is outside that "
                      f"song's window [{match['start']}, {match['end']}) -- verify by hand")
            letter = next_letter(num)
            out_name = f"{num}-{title}-{letter}{os.path.splitext(fn)[1].lower()}"
            shutil.copy2(src, os.path.join(args.out, out_name))
            manifest.append({"file": out_name, "source": name, "orig": fn,
                              "start": start.isoformat(), "end": end.isoformat(), "songs": [num]})
            print(f"  {name}/{fn} -> {out_name}")

    # --- raw / offset-corrected contributors: assign by real-timestamp overlap ---
    def process_raw(name, src_dir, offset_seconds):
        for fn in sorted(os.listdir(src_dir)):
            if not fn.lower().endswith((".mp4", ".mov")):
                continue
            src = os.path.join(src_dir, fn)
            dur = float(ffprobe("format=duration", src))
            if offset_seconds:
                tmp_dst_probe = src  # read offset from original, remux happens after assignment
                ct = ffprobe("format_tags=creation_time", src)
                start = parse_iso(ct) + timedelta(seconds=offset_seconds)
            else:
                ct = ffprobe("format_tags=creation_time", src)
                start = parse_iso(ct)
            end = start + timedelta(seconds=dur)
            overlaps = assign_overlaps(start, end, songs)
            if not overlaps:
                print(f"  NO SONG OVERLAP (skipped): {name}/{fn}  [{start} - {end})")
                continue
            if len(overlaps) == 1:
                s = overlaps[0]
                song_key = s["num"]
                letter = next_letter(song_key)
                out_name = f"{s['num']}-{sanitize(s['title'])}-{letter}{os.path.splitext(fn)[1].lower()}"
                songs_list = [s["num"]]
            else:
                lo, hi = overlaps[0], overlaps[-1]
                song_key = f"{lo['num']}-{hi['num']}"
                letter = next_letter(song_key)
                out_name = (f"{lo['num']}-{hi['num']}-{sanitize(lo['title'])} to "
                            f"{sanitize(hi['title'])}-{letter}{os.path.splitext(fn)[1].lower()}")
                songs_list = [s["num"] for s in overlaps]
                print(f"  BRIDGE clip: {name}/{fn} spans songs {songs_list}")
            dst = os.path.join(args.out, out_name)
            if offset_seconds:
                remux_with_offset(src, dst, offset_seconds)
            else:
                shutil.copy2(src, dst)
            manifest.append({"file": out_name, "source": name, "orig": fn,
                              "start": start.isoformat(), "end": end.isoformat(), "songs": songs_list})
            print(f"  {name}/{fn} -> {out_name}")

    for spec in args.raw:
        name, src_dir = spec.split("=", 1)
        process_raw(name, src_dir, 0)

    for spec in args.offset_corrected:
        name, src_dir, secs = spec.split("=", 2)
        process_raw(name, src_dir, float(secs))

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"songs": [{"num": s["num"], "title": s["title"],
                               "start": s["start"].isoformat(), "end": s["end"].isoformat()}
                              for s in songs],
                   "clips": manifest}, f, indent=2)
    print(f"\nWrote {len(manifest)} clips. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
