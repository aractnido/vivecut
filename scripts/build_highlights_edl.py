r"""build_highlights_edl.py -- pick an edit decision list (EDL) out of the full Give 2026 show's
unified timeline.json: the FULL show, every song kept in full, with dead air trimmed out (not a
compressed ~30min highlights reel -- that was the original design, dropped per direct request:
"we are no longer targeting a full video of 30 minutes, but the full show, without long pauses").

Three nested decisions, driven by signals already extracted in Stage 1 plus two real-editing
conventions confirmed by research (see SKILL.md / session notes -- cuts should follow the song's
own tempo/phrasing, not a fixed clock; a camera shouldn't park on one shot too long):

  1. WHICH WINDOW of each song to keep. The window OPENS at the song's own real first note
     (find_song_onset() -- fine-grained RMS-rise detection on the actual audio, not the coarse
     1-second AudioEnergy trace) and CLOSES wherever sustained quiet begins afterward
     (find_song_end() -- the same RMS-envelope technique, scanning for a sustained drop instead
     of a sustained rise): applause fading, between-song banter, tuning. This keeps the real song
     and trims the dead time around it, without compressing the song itself down to a shorter
     "highlight."
  2. WHERE the cuts land, moment to moment, inside that window. Tempo-aware: this song's own
     BPM (librosa beat-tracking on its actual highlight-window audio, same technique used for
     the single-song beatsync demo earlier in this project) sets a baseline shot length scaled
     from a 120bpm reference, and every candidate cut point snaps to the NEAREST REAL BEAT near
     that nominal length rather than a mechanical fixed-15s clock -- a fast song gets shorter,
     more frequent cuts; a slow one holds longer, same as a human editor would pace it.
  3. WHICH CAMERA to show for that segment. Scored by its own mean person/pose activity
     (extract_people.py) with a zoom/glow quality multiplier and a blackout disqualification.
     Nikon (the only camera guaranteed to cover every chunk) gets a flat bonus so it stays the
     safe default; a stickiness bonus discourages flapping, and a forced-variety rule prevents
     any one non-Nikon camera from holding the shot indefinitely (max dwell scales with tempo
     too, via the chunk length). An ANTICIPATION bonus looks a few seconds PAST the segment
     being scored: if that camera's own activity is about to rise sharply (someone starting to
     walk toward a mic, about to become active), the score is boosted now, so the director is
     already on that camera a few seconds before the action peaks, not reacting after the fact.

GoPro1/GoPro2 candidates prefer the lens/roll-corrected clips (MastersGoPro{1,2}-Corrected,
see apply_lens_correction.py) over the raw fisheye-distorted Masters copies, falling back to
raw if a corrected version isn't available yet for that specific clip.

Nikon's own continuous audio is always the output audio for a song's highlight window,
regardless of which camera's video is showing -- it's the clean FOH-mic master, every other
camera's audio is scratch-reference quality only, and audio never switches when the camera does.

Output: a single JSON EDL (highlights_edl.json), no rendering here -- inspect/adjust the EDL
before spending GPU/ffmpeg time on it.
"""
import json
import tempfile
import os
import subprocess
from datetime import datetime, timedelta

import librosa
import numpy as np

MC = os.environ.get("VIVECUT_SHOW_ROOT", r"C:\path\to\your\show-working-folder")
TIMELINE = os.path.join(MC, "timeline.json")
OUT = os.path.join(MC, "highlights_edl.json")
SCRATCH = os.path.join(tempfile.gettempdir(), "vivecut_beats")

SEG_LEN_REF = 11.0      # baseline shot length at REF_BPM (was 15.0 -- more dynamic pacing)
REF_BPM = 120.0
MIN_SEG, MAX_SEG = 6.0, 18.0
# Nikon/sticky bonuses were originally sized without checking them against real measured
# activity deltas between cameras -- those deltas run ~0.02-0.08 on this footage (see session
# notes), so the old NIKON_BONUS=0.15 + STICKY_BONUS=0.10 (+ up to ANTICIPATE_BONUS=0.20) easily
# outweighed the actual content signal, making Nikon win almost every segment regardless of which
# camera actually had the more active/interesting shot -- confirmed on song 01: GoPro1 had HIGHER
# raw activity than Nikon on segment 30.4-43.0 (0.121 vs 0.098) and still lost. These are now small
# tie-breakers, not auto-wins, so the actual per-segment signal drives the cut.
NIKON_BONUS = 0.05
STICKY_BONUS = 0.04
MAX_DWELL_CHUNKS = 2    # before forcing a look elsewhere (chunk length itself is tempo-scaled)
ANTICIPATE_SEC = 5.0    # how far ahead to look for a coming rise in activity
ANTICIPATE_RISE_RATIO = 1.4   # future mean must exceed current mean by this multiple
ANTICIPATE_BONUS = 0.20
# First-appearance bonus for a camera not yet used in this song -- without this, a slightly
# lower-scoring but still decent angle (Iris/Nova passing by with a phone, GoPro's wide
# coverage) never gets shown at all once Nikon is winning narrowly; real multicam editing samples
# every available angle at least once rather than parking on whichever camera scores highest in a
# vacuum. One-time per camera per song, not a recurring bonus -- it's there to GET a camera its
# first look, not to keep it on screen after that.
NOVELTY_BONUS = 0.12

ORIENTATION = {
    "nikon": "landscape", "gopro1": "landscape", "gopro2": "landscape",
    "iris": "portrait", "nova": "portrait",
}
CORRECTED_DIRS = {
    "gopro1": os.path.join(MC, "MastersGoPro1-Corrected"),
    "gopro2": os.path.join(MC, "MastersGoPro2-Corrected"),
    "iris": os.path.join(MC, "MastersIris-Stabilized"),
    "nova": os.path.join(MC, "MastersNova-Stabilized"),
}


def parse(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean_over(arr, lo, hi):
    """Average a per-second array over [lo,hi) seconds, clamped to array bounds."""
    lo_i, hi_i = max(0, int(lo)), min(len(arr), int(hi) + 1)
    if hi_i <= lo_i:
        return None
    vals = [v for v in arr[lo_i:hi_i] if v is not None]
    return sum(vals) / len(vals) if vals else None


def zoomglow_quality(zg, lo, hi):
    if not zg or not zg.get("trace"):
        return 1.0
    pts = [p.get("score") for p in zg["trace"] if lo <= p["t"] < hi]
    pts = [p for p in pts if p is not None]
    return (sum(pts) / len(pts)) if pts else 1.0


def overlaps_blackout(bo, lo, hi):
    if not bo:
        return False
    for iv in bo.get("blackout_intervals", []):
        if iv[0] < hi and iv[1] > lo:
            return True
    return False


def resolve_clip_path(src, filename):
    if src in CORRECTED_DIRS:
        corrected = os.path.join(CORRECTED_DIRS[src], filename)
        if os.path.exists(corrected):
            return corrected
    return os.path.join(MC, "Masters", filename)


def song_beats(nikon_file, hl_start_local, hl_end_local, song_num):
    """Beat times (seconds, LOCAL to the Nikon file, i.e. already offset by hl_start_local) and
    estimated BPM for this song's own highlight window, via librosa on the real audio -- not
    inferred from the coarse per-second energy trace."""
    os.makedirs(SCRATCH, exist_ok=True)
    wav = os.path.join(SCRATCH, f"{song_num}.wav")
    dur = hl_end_local - hl_start_local
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(hl_start_local), "-i", nikon_file,
                     "-t", str(dur), "-ac", "1", "-ar", "22050", wav], check=True)
    y, sr = librosa.load(wav, sr=22050)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) + hl_start_local
    os.remove(wav)
    bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
    return bpm, list(beat_times)


ONSET_SEARCH_SEC = 20.0   # how far into the song to look for the real start
ONSET_RISE_DB = 15.0      # a transient must rise at least this many dB above the recent noise
                           # floor to count as a real note, not room/mic hiss
ONSET_BASELINE_SEC = 1.5  # "recent noise floor" = the minimum RMS in this many preceding seconds
LEAD_IN_SEC = 0.1         # tiny pad so the very first attack transient isn't clipped by the cut

# Manual per-song onset overrides, song_num -> seconds. find_song_onset()'s RMS-rise detection
# finds the first real NOTE, but for song 09 ("Sweet Caroline") that trims away the "SWEET
# CAROLINE" title-card screen entirely -- the card is already showing at local 0 of the per-song
# split and the user explicitly wants it visible in the final cut, so this starts at the true
# beginning of the split instead of at the detected musical onset.
ONSET_OVERRIDE_SEC = {"09": 0.0}


def find_song_onset(nikon_file):
    """Real audio analysis -- NOT the coarse 1-second-bucketed AudioEnergy trace, which is too
    coarse to tell an individual quiet pickup note from the room silence around it. This computes
    a fine-grained RMS envelope (~23ms resolution via librosa) directly from the song's own audio
    and finds the first point where loudness jumps ONSET_RISE_DB or more above the local noise
    floor -- a genuine transient emerging from near-silence, i.e. the first real note.

    Verified against song 01 ("Summer Of 69") two independent ways before trusting this: (1) the
    RMS envelope by eye shows three individual guitar plucks at ~1.49s/1.90s/2.32s (each decaying
    back toward silence between hits) before the sustained/full-band entrance at ~2.74s: picking
    the loudest contiguous stretch anywhere in the song (the old approach) landed on vocals at
    18s, deep into the song. (2) librosa's own onset_detect (spectral-flux based, a different
    algorithm) independently finds the same first real onset at 1.486s (strength 0.44), after
    two near-zero-strength spurious onsets at 0.046s/0.697s that are just noise floor, not notes.
    This function reproduces that same ~1.49s answer on its own terms, using the RMS-rise
    definition because it's simpler to reason about and tune (one threshold: "how many dB above
    recent quiet counts as a real sound") than onset-strength's unitless scale."""
    SR = 22050
    raw = subprocess.run(["ffmpeg", "-v", "error", "-t", str(ONSET_SEARCH_SEC), "-i", nikon_file,
                           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    if len(y) < SR:
        return 0.0
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    times = librosa.frames_to_time(np.arange(len(db)), sr=SR, hop_length=hop)
    frames_per_sec = SR / hop
    baseline_frames = max(1, int(ONSET_BASELINE_SEC * frames_per_sec))
    for i in range(baseline_frames, len(db)):
        floor = db[max(0, i - baseline_frames):i].min()
        if db[i] - floor >= ONSET_RISE_DB:
            return float(times[i])
    return 0.0


DEAD_AIR_DB_BELOW_TYPICAL = 18.0  # how far below the song's own typical level counts as "dead air"
DEAD_AIR_HOLD_SEC = 5.0           # sustained quiet this long means the song's actually over, not
                                   # just a drum break or a quiet bridge


def find_song_end(nikon_file, onset, orig_dur):
    """Scan forward from the song's real start (find_song_onset()'s result) to find where
    SUSTAINED quiet begins -- applause fading, between-song banter, tuning -- so the kept window
    is the song itself, not however many extra minutes of talk happened to share the same
    continuous Nikon recording. Mirrors find_song_onset()'s RMS-envelope/relative-threshold
    logic, just scanning for a sustained DROP instead of a sustained RISE. Falls back to the
    clip's own full length if no sustained quiet stretch is ever found (a song that runs right up
    against the next song's clip boundary with no gap)."""
    SR = 22050
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(onset), "-i", nikon_file,
                           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    if len(y) < SR:
        return orig_dur
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    times = librosa.frames_to_time(np.arange(len(db)), sr=SR, hop_length=hop) + onset
    frames_per_sec = SR / hop
    hold_frames = max(1, int(DEAD_AIR_HOLD_SEC * frames_per_sec))

    finite = [v for v in db if np.isfinite(v)]
    vals = sorted(finite)
    typical = vals[len(vals) // 2] if vals else 0.0
    thresh = typical - DEAD_AIR_DB_BELOW_TYPICAL

    i = 0
    n = len(db)
    while i < n - hold_frames:
        if np.all(db[i:i + hold_frames] < thresh):
            return float(times[i])
        i += 1
    return orig_dur


def nearest_beat(beats, target, lo_bound):
    """Nearest beat to `target` that's still comfortably past `lo_bound` (never a backward cut)."""
    candidates = [b for b in beats if b > lo_bound + MIN_SEG * 0.5]
    if not candidates:
        return None
    return min(candidates, key=lambda b: abs(b - target))


def main():
    tl = json.load(open(TIMELINE, encoding="utf-8"))
    nikon_clips = sorted(tl["cameras"]["nikon"], key=lambda c: c["real_start_utc"])

    orig_durs = [(parse(c["real_end_utc"]) - parse(c["real_start_utc"])).total_seconds() for c in nikon_clips]

    other_clips = {src: tl["cameras"][src] for src in ("iris", "nova", "gopro1", "gopro2")}

    songs_out = []
    for nc, orig_dur in zip(nikon_clips, orig_durs):
        song_num, title = nc["song_num"], nc["title"]
        nikon_start = parse(nc["real_start_utc"])

        onset = ONSET_OVERRIDE_SEC[song_num] if song_num in ONSET_OVERRIDE_SEC else find_song_onset(nc["file"])
        hl_start_local = onset + LEAD_IN_SEC
        hl_end_local = find_song_end(nc["file"], hl_start_local, orig_dur)
        target_dur = hl_end_local - hl_start_local

        hl_abs_start = nikon_start + timedelta(seconds=hl_start_local)
        hl_abs_end = nikon_start + timedelta(seconds=hl_end_local)

        bpm, beats = song_beats(nc["file"], hl_start_local, hl_end_local, song_num)
        seg_len_song = min(MAX_SEG, max(MIN_SEG, SEG_LEN_REF * (REF_BPM / max(bpm, 1.0))))
        print(f"{song_num} {title}: {bpm:.0f} BPM -> ~{seg_len_song:.1f}s shots, {len(beats)} beats detected")

        nikon_act = load_json(nc.get("activity"))
        nikon_mean = nikon_act["mean"] if nikon_act else None

        segments = []
        prev_cam = None
        dwell = 0
        seen_cams = set()
        t = hl_start_local
        while t < hl_end_local:
            nominal_hi = t + seg_len_song
            seg_hi = nearest_beat(beats, nominal_hi, t)
            if seg_hi is None or seg_hi >= hl_end_local:
                seg_hi = hl_end_local
            seg_lo = t
            abs_lo = nikon_start + timedelta(seconds=seg_lo)
            abs_hi = nikon_start + timedelta(seconds=seg_hi)

            candidates = []  # dicts: score, source, file, lo, hi, + the breakdown that built score

            if nikon_mean is not None:
                m = mean_over(nikon_mean, seg_lo, seg_hi)
                if m is not None:
                    future = mean_over(nikon_mean, seg_hi, seg_hi + ANTICIPATE_SEC)
                    bonus = ANTICIPATE_BONUS if (future and m > 0 and future > m * ANTICIPATE_RISE_RATIO) else 0.0
                    candidates.append({"score": m + NIKON_BONUS + bonus, "source": "nikon",
                                        "file": nc["file"], "lo": seg_lo, "hi": seg_hi,
                                        "activity": m, "quality": 1.0, "nikon_bonus": NIKON_BONUS,
                                        "anticipate_bonus": bonus})

            for src, clips in other_clips.items():
                for c in clips:
                    c_start, c_end = parse(c["start"]), parse(c["end"])
                    # FULL coverage required, not just partial overlap. A candidate that only
                    # partially overlaps [abs_lo, abs_hi) used to get clamped (loc_lo/loc_hi
                    # trimmed to whatever the clip actually has) while the segment's own
                    # abs_start/abs_end kept claiming the FULL nominal window -- silently
                    # mismatching the segment's real-world time span against what its clamped
                    # video content actually shows. Caught via frame-level ground truth (a
                    # karaoke lyric's white->yellow state didn't match between the rendered
                    # video and Nikon's own footage at the "same" moment): a candidate whose own
                    # clip started 2.5s after the window's nominal start got clamped to its own
                    # t=0, but the segment still claimed to start at the nominal (2.5s-earlier)
                    # real time, so the video ran 2.5s ahead of the audio from that point on.
                    if c_start <= abs_lo and c_end >= abs_hi:
                        loc_lo = (abs_lo - c_start).total_seconds()
                        loc_hi = (abs_hi - c_start).total_seconds()
                        bo = load_json(c.get("blackout"))
                        if overlaps_blackout(bo, loc_lo, loc_hi):
                            continue
                        act = load_json(c.get("activity"))
                        if not act or not act.get("mean"):
                            continue
                        m = mean_over(act["mean"], loc_lo, loc_hi)
                        if m is None:
                            continue
                        future = mean_over(act["mean"], loc_hi, loc_hi + ANTICIPATE_SEC)
                        bonus = ANTICIPATE_BONUS if (future and m > 0 and future > m * ANTICIPATE_RISE_RATIO) else 0.0
                        zg = load_json(c.get("zoomglow"))
                        q = zoomglow_quality(zg, loc_lo, loc_hi)
                        full_path = resolve_clip_path(src, c["file"])
                        candidates.append({"score": m * q + bonus, "source": src, "file": full_path,
                                            "lo": loc_lo, "hi": loc_hi, "activity": m, "quality": q,
                                            "nikon_bonus": 0.0, "anticipate_bonus": bonus})

            if not candidates:
                t = seg_hi
                continue

            for cand in candidates:
                cand["sticky_bonus"] = STICKY_BONUS if cand["source"] == prev_cam else 0.0
                cand["novelty_bonus"] = NOVELTY_BONUS if cand["source"] not in seen_cams else 0.0
                cand["total"] = cand["score"] + cand["sticky_bonus"] + cand["novelty_bonus"]
            candidates.sort(key=lambda c: -c["total"])

            if prev_cam != "nikon" and dwell >= MAX_DWELL_CHUNKS:
                alt = [c for c in candidates if c["source"] != prev_cam]
                if alt:
                    candidates = alt

            winner = candidates[0]
            runner_up = candidates[1] if len(candidates) > 1 else None
            src, f, loc_lo, loc_hi, raw_score = (winner["source"], winner["file"], winner["lo"],
                                                  winner["hi"], winner["score"])

            # Human-readable "why this camera, this moment" -- built from the actual bonus
            # breakdown that produced the winning score, not reverse-engineered after the fact,
            # so it stays accurate if the bonus values/logic above ever change. Added per direct
            # request: a raw "score": 0.388 number alone doesn't say WHY a camera won.
            reason_bits = [f"activity {winner['activity']:.3f}"]
            if winner["quality"] != 1.0:
                reason_bits.append(f"zoom/glow quality x{winner['quality']:.2f}")
            if winner["nikon_bonus"]:
                reason_bits.append(f"carries show audio (+{winner['nikon_bonus']:.2f})")
            if winner["anticipate_bonus"]:
                reason_bits.append(f"anticipation -- activity about to rise (+{winner['anticipate_bonus']:.2f})")
            if winner["sticky_bonus"]:
                reason_bits.append(f"stayed on this camera, avoiding flicker (+{winner['sticky_bonus']:.2f})")
            if winner["novelty_bonus"]:
                reason_bits.append(f"first look at this camera this song (+{winner['novelty_bonus']:.2f})")
            reason = f"{winner['source']}: " + ", ".join(reason_bits) + f" = {winner['total']:.3f} total"
            if runner_up:
                reason += (f"; beat {runner_up['source']} ({runner_up['total']:.3f}) by "
                           f"{winner['total'] - runner_up['total']:.3f}")
            else:
                reason += " (only candidate covering this window)"

            # QA guard: the clip-local span this segment will actually show MUST equal the
            # real-world span it claims to cover, or render_song.py's duration-based audio
            # timeline silently desyncs from the video from here on (see the full-coverage-only
            # fix above -- this assertion is the tripwire for if that invariant ever breaks
            # again, instead of needing another multi-hour frame-by-frame manual hunt to notice).
            real_span = (abs_hi - abs_lo).total_seconds()
            clip_span = loc_hi - loc_lo
            assert abs(real_span - clip_span) < 0.05, (
                f"EDL span mismatch song {song_num} src={src} file={f}: "
                f"real-world span {real_span:.3f}s vs clip span {clip_span:.3f}s -- "
                f"this segment's video would NOT match its claimed real-world time window"
            )
            segments.append({
                "abs_start": abs_lo.isoformat(), "abs_end": abs_hi.isoformat(),
                "source": src, "clip_file": f, "orientation": ORIENTATION[src],
                "local_start": round(loc_lo, 2), "local_end": round(loc_hi, 2),
                "score": round(raw_score, 3),
                "reason": reason,
            })
            dwell = dwell + 1 if src == prev_cam else 1
            prev_cam = src
            seen_cams.add(src)
            t = seg_hi

        songs_out.append({
            "song_num": song_num, "title": title,
            "nikon_file": nc["file"], "bpm": round(bpm, 1),
            "highlight_abs_start": hl_abs_start.isoformat(),
            "highlight_abs_end": hl_abs_end.isoformat(),
            "highlight_local_start": hl_start_local, "highlight_local_end": hl_end_local,
            "kept_duration": round(target_dur, 1), "orig_duration": round(orig_dur, 1),
            "segments": segments,
        })

    # Write atomically (temp file + os.replace) -- render_song.py invocations for other songs may
    # be reading this same file concurrently while this one's being regenerated mid-batch.
    tmp_out = OUT + ".tmp"
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump({"songs": songs_out}, f, indent=2)
    os.replace(tmp_out, OUT)

    total = sum(s["kept_duration"] for s in songs_out)
    print(f"songs={len(songs_out)}  total kept duration={total/60:.1f} min (full songs, dead air trimmed)")
    cam_counts, cut_counts = {}, {}
    for s in songs_out:
        cut_counts[s["song_num"]] = len(s["segments"])
        for seg in s["segments"]:
            cam_counts[seg["source"]] = cam_counts.get(seg["source"], 0) + (
                parse(seg["abs_end"]) - parse(seg["abs_start"])
            ).total_seconds()
    for src, secs in sorted(cam_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {secs/60:.1f} min on screen")
    print(f"total cuts: {sum(cut_counts.values())}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
