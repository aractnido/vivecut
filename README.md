# vivecut

An AI director for multi-camera live-show footage. Give it a Nikon/camcorder master recording
plus however many phone and action-cam clips your friends shot, and it builds a unified
real-world timeline, analyzes performer movement and shot quality on every camera, and renders
a full multicam edit -- deciding shot-by-shot which camera is actually the most interesting one
to be on, instead of a human scrubbing through five angles of a two-hour show by hand.

Built as a [Claude Code](https://claude.com/claude-code) skill. Drop this folder into
`.claude/skills/` (or your own skills directory) and point Claude at your raw footage.

## What it does

1. **Builds a unified timeline.** Every camera's clips get matched to real-world UTC time via
   audio cross-correlation (GCC-PHAT) against the master recording, not by trusting each
   camera's own clock -- phone clocks and GoPro clocks drift, lie, or reset.
2. **Scores every camera, every moment.** Pose/movement activity, audio energy, shot quality
   (zoom/glow detection, blackout detection), with a few real-editing heuristics baked in:
   don't flicker between cameras too fast, give every camera at least one look per song, and
   anticipate a camera that's *about* to get interesting a few seconds before it peaks.
3. **Cuts on the beat.** Shot length is tempo-aware (via `librosa` beat tracking on the song's
   own audio), not a fixed clock -- fast songs get faster cuts, slow songs hold longer.
4. **Renders the full show.** 4K canvas, per-source treatments (landscape scale-to-fill,
   portrait blur-fill for phones, centered blur-fill for action cams), GPU-accelerated where it's
   been verified pixel-identical to the CPU path, with a full provenance JSON sidecar explaining
   *why* the director picked each cut.

See `SKILL.md` for the full pipeline (one script per stage), the hard-won lessons from building
this against real footage (clock drift, aliased audio correlation, frame-rounding drift, GPU
pixel-discrepancy dead ends), and the actual scoring math.

## Status

Built and proven end-to-end on a real 22-song, 5-camera show. The per-show director/render
scripts (`build_highlights_edl.py`, `render_song.py`) still hardcode some show-specific
assumptions (a fixed show-root path via `VIVECUT_SHOW_ROOT`, a 4-role band/camera layout) --
treat them as a working reference implementation to adapt, not a drop-in CLI tool yet.
