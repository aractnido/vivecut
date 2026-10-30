"""
make_song_windows.py

Builds the song-windows CSV that build_masters.py needs, from a camera folder whose
files are ALREADY one-file-per-song with a correct real-world clock (e.g. a GoPro set to
auto-chapter, or any camera a human already split/labeled per song). Output columns:
song,creation_time,duration -- "song" is the file's own basename (NN-Title...), so
build_masters.py can recover both the song number and a human-readable title from it.

Usage:
  py make_song_windows.py <camera_dir> <out.csv> [--ext mp4,mov]
"""
import argparse, csv, os, subprocess, sys


def ffprobe(field, path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", field,
                         "-of", "default=nw=1:nk=1", path],
                        capture_output=True, text=True)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camera_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--ext", default="mp4,mov")
    args = ap.parse_args()
    exts = tuple("." + e.strip().lower() for e in args.ext.split(","))

    rows = []
    for fn in sorted(os.listdir(args.camera_dir)):
        if not fn.lower().endswith(exts):
            continue
        path = os.path.join(args.camera_dir, fn)
        ct = ffprobe("format_tags=creation_time", path)
        dur = ffprobe("format=duration", path)
        if not ct or not dur:
            print(f"  SKIP (no creation_time/duration): {fn}", file=sys.stderr)
            continue
        rows.append((fn, ct, dur))
        print(f"  {fn}  {ct}  {dur}s")

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["song", "creation_time", "duration"])
        for fn, ct, dur in rows:
            w.writerow([fn, ct, dur])
    print(f"\nWrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
