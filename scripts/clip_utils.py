"""Shared helper for any script in this skill that scans a Masters folder for video clips.
Pulled out of audio_energy.py, extract_people.py, extract_depth.py, detect_shakiness.py, and
apply_stabilization.py, which each carried a byte-for-byte copy -- a fix to clip discovery
previously had to be hand-applied in all five places, and missing one would leave that script
silently scanning a different clip set than the others in the same pipeline run."""
import glob
import os

CLIP_GLOBS = ("*.mp4", "*.mov", "*.MP4", "*.MOV")


def find_clips(masters_dir):
    seen = {}
    for pattern in CLIP_GLOBS:
        for f in glob.glob(os.path.join(masters_dir, pattern)):
            seen[os.path.normcase(os.path.abspath(f))] = f
    return sorted(seen.values())
