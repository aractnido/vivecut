r"""precise_audio_sync.py -- measure and correct each individual clip's real-world start time
against the Nikon master, via GCC-PHAT audio cross-correlation at full (44.1kHz) resolution.

Why per-clip, not one flat correction per camera: a flat constant (this script's own earlier
pass, see SKILL.md) gets a camera's AVERAGE error right, but individual clips vary around that
average by up to ~100-200ms -- not visible in a loose "does this look synced" check, but very
audible as lip-sync lag/lead once you're actually watching someone sing.

**The search window MUST be constrained to a margin around the existing metadata offset, not the
whole song.** First version of this script searched the entire song for every clip and got
burned: several clips got wildly wrong deltas (+145s, +193s, +284s...) at HIGH confidence
(z=15-21) because the song has repeated musical phrases (verse/chorus structure) -- an
unconstrained cross-correlation happily locks onto the wrong repeat of a near-identical riff
elsewhere in the song, and the z-score alone doesn't warn you, since the match really is strong,
just strong at the wrong place. Caught this by building an independent, non-audio ground truth:
extracting frames around the suspect timestamp from both Nikon and the clip, finding a moment
with on-screen karaoke lyrics (white = not yet sung, yellow = currently singing) visible in BOTH
cameras, and comparing exactly when that white->yellow transition happens in each. That frame-
verified true offset landed within ~30-100ms of both the existing metadata AND an earlier
narrow-window GCC-PHAT pass -- confirming the narrow/constrained search is correct and the
full-song search was the actual bug, not the earlier correction. SEARCH_MARGIN_SEC below is sized
to comfortably cover any realistic residual error (GoPro's was ~1s, Iris's ~0.1s) while staying
well short of where this song's phrases repeat.

Usage:
  py precise_audio_sync.py <timeline.json> <camera> <masters_dir> [--apply] [--min-z 8.0] [--margin 20.0]

Without --apply: just reports each clip's current metadata offset, the measured true offset, the
delta, and the confidence (z-score) -- inspect before trusting. With --apply: writes the measured
correction into timeline.json for every clip whose z-score clears --min-z (default 8.0, same
confidence floor used elsewhere in this project's audio-sync work); clips below that floor are
left untouched and listed as low-confidence, since a bad correction is worse than no correction.
"""
import glob
import json
import os
import subprocess
import sys
from datetime import timedelta

import numpy as np

from audio_sync_utils import SR, aud, fmt, gcc_phat, parse

SEARCH_MARGIN_SEC = 20.0  # how far from the existing metadata offset to search -- wide enough to
                          # catch a real error, narrow enough to not reach the song's own repeats
SANITY_MAX_DELTA = 2.0   # every genuine measured delta seen on real clips so far is sub-1-second
                          # (even a high z-score doesn't mean the match is right -- two GoPro1
                          # clips landed at -9.0s/+8.2s with z=33/56, suspiciously close to this
                          # song's own repeat-phrase spacing; a strong peak at the WRONG repeat is
                          # still a strong peak). Anything bigger is almost certainly a bad match
                          # (e.g. a clip whose metadata offset is so far negative/past-the-song that
                          # clamping the search window to the song's own bounds forced a spurious
                          # match, or a strong-but-wrong repeat) -- flag it, don't trust z alone


def main():
    args = sys.argv[1:]
    timeline_path, camera, masters_dir = args[0], args[1], args[2]
    apply = "--apply" in args
    min_z = 8.0
    if "--min-z" in args:
        min_z = float(args[args.index("--min-z") + 1])
    margin = SEARCH_MARGIN_SEC
    if "--margin" in args:
        margin = float(args[args.index("--margin") + 1])

    tl = json.load(open(timeline_path, encoding="utf-8"))
    nikon_by_song = {c["song_num"]: c for c in tl["cameras"]["nikon"]}

    results = []
    for c in tl["cameras"][camera]:
        songs = c.get("songs", [])
        if not songs or songs[0] not in nikon_by_song:
            continue
        nc = nikon_by_song[songs[0]]
        n_start = parse(nc["real_start_utc"])
        c_start = parse(c["start"])
        meta_offset = (c_start - n_start).total_seconds()
        dur = (parse(c["end"]) - c_start).total_seconds()

        path = os.path.join(masters_dir, c["file"])
        if not os.path.exists(path):
            matches = glob.glob(os.path.join(masters_dir, c["file"]))
            if not matches:
                print(f"SKIP (file not found): {c['file']}")
                continue
            path = matches[0]

        sig = aud(path, 0, min(dur, 40))
        ref_dur_cmd = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                       "-of", "csv=p=0", nc["file"]], capture_output=True, text=True)
        song_dur = float(ref_dur_cmd.stdout.strip())
        ref_start = max(0.0, meta_offset - margin)
        ref_dur = min(song_dur - ref_start, min(dur, 40) + 2 * margin)
        ref = aud(nc["file"], ref_start, ref_dur)
        lag, z = gcc_phat(sig, ref)
        measured = ref_start - lag
        delta = measured - meta_offset
        # A delta landing right at +/-margin (within a few ms) means the true match is OUTSIDE
        # the search window, not that this is really the best alignment -- a confident-looking
        # but meaningless boundary artifact, the GCC-PHAT equivalent of "search cut off too
        # soon." Needs a wider or re-targeted search, not blind trust just because z is high.
        at_boundary = abs(abs(delta) - margin) < 0.05
        suspicious = at_boundary or abs(delta) > SANITY_MAX_DELTA
        results.append((c["file"], meta_offset, measured, delta, z, suspicious))
        if at_boundary:
            flag = "BOUNDARY-HIT (search window too narrow)"
        elif suspicious:
            flag = f"SUSPICIOUS (|delta|>{SANITY_MAX_DELTA}s)"
        else:
            flag = "OK" if z >= min_z else "LOW-CONFIDENCE"
        print(f"{c['file']}: meta={meta_offset:8.3f}  measured={measured:8.3f}  "
              f"delta={delta:+7.3f}  z={z:5.1f}  [{flag}]")

    confident = [r for r in results if r[4] >= min_z and not r[5]]
    low_conf = [r for r in results if r[4] < min_z or r[5]]
    deltas = [r[3] for r in confident]
    if deltas:
        print(f"\n{len(confident)}/{len(results)} clips confident (z>={min_z}); "
              f"delta range {min(deltas):+.3f} to {max(deltas):+.3f}, mean {sum(deltas)/len(deltas):+.3f}")
    if low_conf:
        print(f"{len(low_conf)} clip(s) below confidence floor, left uncorrected if --apply: "
              f"{[r[0] for r in low_conf]}")

    if apply:
        by_file = {r[0]: r[3] for r in confident}
        n = 0
        for c in tl["cameras"][camera]:
            if c["file"] in by_file:
                delta = by_file[c["file"]]
                c["start"] = fmt(parse(c["start"]) + timedelta(seconds=delta))
                c["end"] = fmt(parse(c["end"]) + timedelta(seconds=delta))
                n += 1
        json.dump(tl, open(timeline_path, "w", encoding="utf-8"), indent=2)
        print(f"\napplied precise per-clip corrections to {n} {camera} clips in {timeline_path}")


if __name__ == "__main__":
    main()
