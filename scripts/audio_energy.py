r"""audio_energy.py -- Stage 1 of ai-director: per-second loudness (dB) and note-attack density
("flux") for every clip in a Masters folder.

Same math as stage-movement-analysis's audio_energy.py (RMS dB per second block, plus positive
spectral flux from an STFT as a note-attack-density proxy), generalized to iterate an arbitrary
Masters folder's clips (any contributor, any extension) instead of a fixed Songs-4K folder of
one-file-per-song Nikon output. Each clip's OWN audio track is used, so a bridge clip or a
zoomed-in phone clip gets a curve that matches ITS OWN audio, not the show's "true" audio --
cross-referencing which camera's audio is actually the best source for a moment is a Stage 2
question (clip_override / pick_truth_audio territory), not this script's job.

Output per clip, in <out_dir>: <clip-basename>-audioenergy.json
  sr, seconds, db (list), flux (list)

Usage:
  py audio_energy.py <masters_dir> <out_dir>
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.signal import stft

from clip_utils import find_clips

SR = 16000




def energy_for_clip(path):
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", path, "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    secs = len(x) // SR
    if secs < 1:
        return None
    blocks = x[: secs * SR].reshape(secs, SR)
    db = 20 * np.log10(np.sqrt((blocks ** 2).mean(axis=1)) + 1e-9)

    _, t, Z = stft(x, fs=SR, nperseg=1024, noverlap=512)
    mag = np.log1p(np.abs(Z) * 50)
    fl = np.maximum(0, np.diff(mag, axis=1)).mean(axis=0)
    tt = t[1:]
    flux = np.array([fl[(tt >= s) & (tt < s + 1)].mean() if np.any((tt >= s) & (tt < s + 1)) else 0.0
                      for s in range(secs)])
    return db.astype(np.float32), flux.astype(np.float32)


def main():
    masters, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    clips = find_clips(masters)
    print(f"{len(clips)} clips")
    t0 = time.time()
    for i, path in enumerate(clips, 1):
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, base + "-audioenergy.json")
        if os.path.exists(out_path):
            print(f"[{i}/{len(clips)}] skip {base}", flush=True)
            continue
        try:
            result = energy_for_clip(path)
            if result is None:
                print(f"[{i}/{len(clips)}] {base}: too short, skipped", flush=True)
                continue
            db, flux = result
            with open(out_path, "w") as fh:
                json.dump({"sr": SR, "seconds": len(db), "db": db.tolist(), "flux": flux.tolist()}, fh)
            print(f"[{i}/{len(clips)}] {base}: {len(db)}s  loudness {db.mean():.1f} dB  "
                  f"flux {flux.mean():.4f}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(clips)}] {base}: EXCEPTION {e!r}", flush=True)
    print(f"audio energy done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
