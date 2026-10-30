r"""render_song.py -- render ONE song's highlight segment from highlights_edl.json, for smoke
testing before committing to all 22 + final concat (per the project's own "review before final
assembly" rule -- past automated-checks-only edits have shipped real bugs).

Usage: python render_song.py <song_num, e.g. 01>

Output canvas is true 4K (3840x2160) -- every camera in this show (Nikon, GoPro1/2, Iris,
Nova) shoots native 4K, so nothing needs upscaling to fill it.

Pipeline per song:
  1. each EDL segment -> its own normalized 3840x2160/30fps/yuv420p clip, no audio. Three
     per-source treatments:
       - Nikon (and any other full-4K landscape source): scale-to-fill + center crop, same as
         before, just at 4K instead of 1080p.
       - Iris/Nova (portrait phones): blurred scaled-to-fill background + sharp
         centered foreground at full output height, so vertical footage fills the frame
         without pillarboxing or being mistaken for picture-in-picture.
       - GoPro1/GoPro2: after lens correction these are NOT full 4K anymore (the undistort/
         rotate/crop needed to fix the fisheye eats into the frame -- see
         apply_lens_correction.py), so scaling them UP to fill 3840x2160 would mean visibly
         upscaling already-mediocre GoPro glass even further. Instead: downscale to 2560x1440
         ("2K"), center it at that native-ish size (no upscaling), and fill the remaining
         border with a blurred, scaled-to-fill copy of the same shot -- same blur-fill
         technique as the portrait treatment, just applied to a landscape source that's
         smaller than the output canvas instead of a different aspect ratio.
  2. concat demuxer (stream copy -- all segments share identical encode params) -> one
     video-only file for the song
  3. Nikon's own master audio, trimmed to the same highlight window, is the ONLY audio used
     regardless of which camera's video is showing (Nikon is the clean FOH mic; every other
     camera is scratch-reference audio quality only)
  4. song-title lower-left banner (drawtext, textfile= so apostrophes in titles like
     "Sweet Child O' Mine" need no escaping) burned in while muxing video+audio together

All encodes use NVENC (h264_nvenc), not libx264 -- CPU software encoding of 4K is the same real
bottleneck documented elsewhere in this project (see apply_lens_correction_gpu.py), no reason to
pay it again here.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

MC = os.environ.get("VIVECUT_SHOW_ROOT", r"C:\path\to\your\show-working-folder")
EDL = os.path.join(MC, "highlights_edl.json")
OUT_DIR = os.path.join(MC, "Preview", "highlights")
SCRATCH = os.path.join(MC, "scratch", "highlight_render")
FONT = "C:/Windows/Fonts/segoeuib.ttf"

OUT_W, OUT_H = 3840, 2160
GOPRO_SOURCES = ("gopro1", "gopro2")
# AV1 via NVENC (av1_nvenc, 8th-gen/Ada+ only), matching this project's own established final-output
# standard in concert-video-pipeline's 04-Publish-Video.ps1 (preset p7, cq 23, uncapped vbr) -- AV1
# compresses meaningfully better than H.264 at equal quality (measured: 42.9% smaller than this
# script's earlier h264_nvenc cq18 output, re-encoding the same song). 10-bit (yuv420p10le) matches
# Nikon's own native source depth (its Songs-4K master is already yuv420p10le) -- explicit decision to
# preserve full archive fidelity even though the eventual YouTube upload target downconverts everything
# to 8-bit BT.709 limited-range on ingest regardless; the archive master shouldn't be the one to throw
# that precision away. An earlier 10-bit render in this same project session came out unplayable with
# just "-pix_fmt yuv420p10le" alone -- fixed here by also pinning color_range/primaries/transfer/
# colorspace explicitly (bt709 limited/tv range), the same combination proven to play correctly in a
# demo this session (see SKILL.md) -- don't drop these extra tags thinking pix_fmt alone is enough.
NVENC = ["-c:v", "av1_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", "23", "-b:v", "0",
         "-pix_fmt", "yuv420p10le", "-color_range", "tv", "-color_primaries", "bt709",
         "-color_trc", "bt709", "-colorspace", "bt709"]

LANDSCAPE_VF = f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},setsar=1,fps=30,format=yuv420p10le"
# GPU decode+scale for the landscape (Nikon) treatment -- verified pixel-identical to LANDSCAPE_VF
# on a real segment (mean/max abs diff both 0.0) while cutting wall-clock time by ~42% (17.3s ->
# 10.1s on an 8s segment), by moving decode onto NVDEC and the scale onto scale_cuda instead of
# CPU libswscale. No GPU equivalent of `crop` exists in this ffmpeg build, so that one step still
# needs a hwdownload first; everything before it (the expensive decode+scale) stays on the GPU.
# NOT applied to PORTRAIT_FC/GOPRO_FC below -- their blur-fill chain hits a real but unexplained
# ~1/255 mean pixel drift once gblur and overlay_cuda are involved (two targeted fixes didn't
# close it), so those keep the proven CPU path rather than ship an unverified discrepancy.
LANDSCAPE_VF_GPU = (
    f"fps=30,scale_cuda={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase:format=p010le,"
    f"hwdownload,format=p010le,crop={OUT_W}:{OUT_H},setsar=1,format=yuv420p10le"
)
PORTRAIT_FC = (
    f"[0:v]split=2[bg][fg];"
    f"[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},gblur=sigma=20[bgb];"
    f"[fg]scale=-2:{OUT_H}[fgs];"
    f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2:format=auto,fps=30,format=yuv420p10le[v]"
)
# GoPro-specific: downscale to 2K and center it in the 4K canvas (no upscaling of already-soft
# GoPro glass), blur-fill the border -- see module docstring for why this differs from LANDSCAPE_VF.
GOPRO_FC = (
    f"[0:v]split=2[bg][fg];"
    f"[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},gblur=sigma=20[bgb];"
    f"[fg]scale=2560:1440[fgs];"
    f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2:format=auto,fps=30,format=yuv420p10le[v]"
)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FFMPEG FAILED:", " ".join(cmd))
        print(r.stderr[-3000:])
        sys.exit(1)


def esc_path(p):
    return p.replace("\\", "/").replace(":", "\\:")


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(r.stdout.strip())


def treatment_for(seg):
    if seg["source"] in GOPRO_SOURCES:
        return "gopro-2k-centered-blurfill"
    return "scale-to-fill-crop" if seg["orientation"] == "landscape" else "portrait-blurfill"


def write_provenance(final_out, song, edl_path):
    """Sidecar <output>.json documenting every cut in the final render: WHERE it falls in the
    output's own timeline, WHAT it's made of (source camera, exact clip file, exact trim in/out,
    the real-world UTC window it corresponds to), and WHY the director picked it -- both the raw
    "director_score" number and a human-readable "director_reason" string built from the actual
    bonus breakdown (activity, zoom/glow quality, audio-track/sticky/novelty/anticipation bonuses,
    and the margin over the runner-up camera; see build_highlights_edl.py for how it's computed).
    This is the standing record of transitions/provenance/decision-making for every rendered
    video from here on, not a one-off for this song."""
    cum = 0.0
    transitions = []
    for i, seg in enumerate(song["segments"]):
        dur = seg["local_end"] - seg["local_start"]
        transitions.append({
            "index": i,
            "render_time_start": round(cum, 3),
            "render_time_end": round(cum + dur, 3),
            "duration": round(dur, 3),
            "source": seg["source"],
            "treatment": treatment_for(seg),
            "clip_file": seg["clip_file"],
            "clip_local_start": seg["local_start"],
            "clip_local_end": seg["local_end"],
            "real_world_start": seg["abs_start"],
            "real_world_end": seg["abs_end"],
            "director_score": seg["score"],
            "director_reason": seg.get("reason", "n/a (rendered before the reason field existed)"),
        })
        cum += dur

    doc = {
        "video": os.path.basename(final_out),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_edl": edl_path,
        "song_num": song["song_num"],
        "title": song["title"],
        "bpm": song["bpm"],
        "audio_source": "nikon master (continuous, unchanged across all cuts)",
        "nikon_file": song["nikon_file"],
        "highlight_real_world_start": song["highlight_abs_start"],
        "highlight_real_world_end": song["highlight_abs_end"],
        "total_cuts": len(song["segments"]),
        "transitions": transitions,
    }
    out_json = final_out + ".json"
    json.dump(doc, open(out_json, "w", encoding="utf-8"), indent=2)
    print("wrote", out_json)


def main():
    song_num = sys.argv[1]
    edl = json.load(open(EDL, encoding="utf-8"))
    song = next(s for s in edl["songs"] if s["song_num"] == song_num)

    work = os.path.join(SCRATCH, song_num)
    os.makedirs(work, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    seg_files = []
    for i, seg in enumerate(song["segments"]):
        dur = seg["local_end"] - seg["local_start"]
        out = os.path.join(work, f"seg_{i:02d}.mp4")
        # -ss AFTER -i (not before) -- accurate/frame-exact seeking, not fast keyframe-snap
        # seeking. This matters a lot here: several source clips (e.g. Iris's phone footage)
        # have ~1-second GOPs, so a fast "-ss before -i" seek can land the cut up to ~1s before
        # the real target -- exactly the kind of error that survives no matter how precisely the
        # offset itself was calculated upstream (found this the hard way chasing what looked like
        # a sync bug that kept not going away even as the measured offsets got more accurate).
        base = ["ffmpeg", "-y", "-i", seg["clip_file"], "-ss", str(seg["local_start"]), "-t", str(dur)]
        if seg["source"] in GOPRO_SOURCES:
            cmd = base + ["-filter_complex", GOPRO_FC, "-map", "[v]", "-an"] + NVENC + [out]
        elif seg["orientation"] == "landscape":
            # hwaccel flags must precede -i, so this path builds its own base rather than reusing
            # the shared one above -- -ss still comes after -i for accurate/frame-exact seeking.
            gpu_base = ["ffmpeg", "-y", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                        "-i", seg["clip_file"], "-ss", str(seg["local_start"]), "-t", str(dur)]
            cmd = gpu_base + ["-vf", LANDSCAPE_VF_GPU, "-an"] + NVENC + [out]
        else:
            cmd = base + ["-filter_complex", PORTRAIT_FC, "-map", "[v]", "-an"] + NVENC + [out]
        print(f"[{i+1}/{len(song['segments'])}] {seg['source']} {seg['local_start']:.1f}-{seg['local_end']:.1f}")
        run(cmd)
        seg_files.append(out)

    list_path = os.path.join(work, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for sf in seg_files:
            f.write(f"file '{sf.replace(chr(92), '/')}'\n")
    video_noaudio = os.path.join(work, "video_noaudio.mp4")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", video_noaudio])

    hl_start = song["highlight_local_start"]
    hl_dur_nominal = song["highlight_local_end"] - hl_start
    # Trim audio to the VIDEO's own real, measured duration, not the nominal EDL duration.
    # Every per-segment cut uses frame-accurate seeking (see the -ss-after-i note above), but
    # "accurate" still means "nearest frame" -- a segment whose nominal length isn't an exact
    # multiple of one frame (33.4ms at 29.97fps) rounds slightly, and those roundings don't
    # cancel out over a whole song. Measured on song 03: 25 segments, net -43ms by the end (video
    # running behind audio) -- small, but above the ~45ms threshold where lip-sync drift starts
    # reading as "off," and it hit EVERY song/camera since the bug is in this cut-and-concat
    # step, not in any one camera's calibration (caught only because the user noticed Nikon's
    # OWN footage looked a little off too, which ruled out a camera-specific cause). Matching
    # audio length to the real video length keeps the whole song's A/V duration equal instead of
    # inheriting the EDL's nominal number, which per-segment rounding never actually delivers.
    video_dur = probe_duration(video_noaudio)
    hl_dur = video_dur
    print(f"audio trim: nominal EDL duration {hl_dur_nominal:.3f}s, actual concatenated video "
          f"duration {video_dur:.3f}s (drift {video_dur - hl_dur_nominal:+.3f}s) -- using actual")
    audio_path = os.path.join(work, "audio.aac")
    run(["ffmpeg", "-y", "-ss", str(hl_start), "-i", song["nikon_file"], "-t", str(hl_dur),
         "-vn", "-c:a", "aac", "-b:a", "320k", audio_path])

    title_txt = os.path.join(work, "title.txt")
    with open(title_txt, "w", encoding="utf-8") as f:
        f.write(f"{song['song_num']} - {song['title']}")

    # Title banner only for the first few seconds (fades out over the last second) -- it was
    # burning in for the whole song before, which reads as a lower-third graphic left on screen
    # by mistake, not an intentional title card.
    TITLE_HOLD = 2.0
    TITLE_FADE = 0.5
    # fontsize/box/margins doubled vs. the old 1080p values to read the same proportionally at 4K.
    drawtext = (
        f"drawtext=fontfile='{esc_path(FONT)}':textfile='{esc_path(title_txt)}':"
        "fontcolor=white:fontsize=96:box=1:boxcolor=black@0.45:boxborderw=28:x=80:y=h-th-80:"
        f"enable='lt(t,{TITLE_HOLD + TITLE_FADE})':"
        f"alpha='if(lt(t,{TITLE_HOLD}),1,max(0,1-(t-{TITLE_HOLD})/{TITLE_FADE}))'"
    )
    safe_title = song["title"].replace("/", "-").replace("?", "").replace(":", "")
    final_out = os.path.join(OUT_DIR, f"{song['song_num']}-{safe_title}.mp4")
    run(["ffmpeg", "-y", "-i", video_noaudio, "-i", audio_path,
         "-vf", drawtext, "-map", "0:v", "-map", "1:a"] + NVENC + ["-c:a", "copy", "-shortest", final_out])

    print("wrote", final_out)
    write_provenance(final_out, song, EDL)


if __name__ == "__main__":
    main()
