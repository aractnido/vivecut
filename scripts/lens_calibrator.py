r"""lens_calibrator.py -- interactive GUI to dial in barrel-distortion correction (k1/k2) and
roll (rotation) on a GoPro frame, live, instead of generating one static comparison image per
guess. Drag the sliders, watch the preview update, hit Save when it looks right.

Usage:
  py lens_calibrator.py <path_to_frame.png>

  # resume from known-good geometry on the ORIGINAL (not pre-corrected) frame -- sliders start
  # at these values instead of the usual first-guess defaults, so you don't have to re-drag
  # k1/k2/rotation/crop back to where they already were just to tweak shadows/brightness/blacks:
  py lens_calibrator.py <path_to_frame.png> k1 k2 angle_deg crop_top crop_right crop_bottom crop_left

Controls:
  k1, k2         -- radial distortion correction (negative = un-fisheye/barrel correct)
  rotation (deg) -- roll correction (positive = clockwise)
  shadows        -- lift dark regions only (0 to 1), fades to nothing in already-bright areas
                    so stage lighting highlights don't get washed out
  brightness     -- flat global exposure shift (-1 darker to 1 brighter), affects everything
                    equally including highlights -- unlike shadows, this CAN blow out or crush
  blacks         -- black-point control (-1 to 1): negative crushes the darkest tones toward
                    pure black (more contrast/punch), positive lifts the black point (flatter,
                    hazier blacks) -- same direction convention as Lightroom's Blacks slider
  grid checkbox  -- overlay a reference grid
  Save button    -- writes a full-resolution PNG next to the input, named with the exact
                    parameters used, and prints the equivalent ffmpeg filter string so the
                    chosen settings can be dropped straight into the production pipeline
                    (geometry only -- tone adjustments have no ffmpeg-filter equivalent
                    printed, see apply_tone's docstring).
"""
import os
import re
import sys

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

PREVIEW_MAX_DIM = 1100


def build_remap(w, h, k1, k2):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    nx = (xs - cx) / cx
    ny = (ys - cy) / cy
    r2 = nx * nx + ny * ny
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    factor = np.clip(factor, 0.05, None)
    src_nx = nx * factor
    src_ny = ny * factor
    map_x = src_nx * cx + cx
    map_y = src_ny * cy + cy
    return map_x, map_y


def apply_tone(img, shadow=0.0, brightness=0.0, blacks=0.0):
    """Three independent tone controls, all operating on the L channel only (LAB, so hue/
    saturation of the stage lighting is untouched), composited in one color-space round-trip
    for live-preview speed. Order applied: blacks (sets the base black point/contrast) ->
    brightness (flat global shift) -> shadows (local lift, computed AFTER the other two so it's
    still only acting on whatever's dark in the final result, not re-darkened by a later step).

    - blacks: black-point remap, -1 (crush -- darkest tones pulled toward 0, more contrast) to
      1 (lift -- darkest tones pulled up toward gray, flatter/hazier). Same convention as
      Lightroom's Blacks slider.
    - brightness: flat additive shift, -1 (darker) to 1 (brighter). Affects EVERYTHING equally,
      including highlights -- unlike shadows below, this can blow out or crush.
    - shadow: 0 (no change) to 1 (strong lift), strongest at L=0 and fading to ~0 by L=255 --
      a flat brightness bump would also blow out this room's stage lighting (otherwise dim,
      lit almost entirely by a handful of colored stage lights), so this only touches dark
      regions.

    No ffmpeg filter reproduces this exact combination, so unlike k1/k2/rotation/crop none of
    these three are in the printed "equivalent ffmpeg filter" string -- apply_lens_correction.py's
    own --shadow/--brightness/--blacks flags are the only place they're reproduced for real.
    """
    if shadow == 0 and brightness == 0 and blacks == 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float32)

    if blacks > 0:
        bp = blacks * 60.0
        l_f = bp + l_f * (255.0 - bp) / 255.0
    elif blacks < 0:
        gamma = 1.0 + (-blacks) * 1.5
        l_f = 255.0 * np.clip(l_f / 255.0, 0, None) ** gamma

    if brightness != 0:
        l_f = l_f + brightness * 100.0
    l_f = np.clip(l_f, 0, 255)

    if shadow > 0:
        lift = shadow * 90.0 * ((255.0 - l_f) / 255.0) ** 2
        l_f = np.clip(l_f + lift, 0, 255)

    lab = cv2.merge([l_f.astype(np.uint8), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_transform(img, k1, k2, angle_deg):
    h, w = img.shape[:2]
    map_x, map_y = build_remap(w, h, k1, k2)
    undistorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)
    rotated = cv2.warpAffine(undistorted, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated


def crop_box(w, h, top_frac, right_frac, bottom_frac, left_frac, target_ratio, lock_ratio=True):
    """All four sides are independently user-controlled. If lock_ratio is on, whatever box
    that produces gets ONE further centered trim (on whichever axis is oversized) to land
    back on the original aspect ratio -- shared evenly rather than dumped onto any single
    side, so all four sliders stay meaningful instead of two of them being silently overridden."""
    x0, x1 = int(w * left_frac), w - int(w * right_frac)
    y0, y1 = int(h * top_frac), h - int(h * bottom_frac)
    W1, H1 = x1 - x0, y1 - y0
    if W1 <= 1 or H1 <= 1:
        return 0, 0, w, h
    if lock_ratio:
        if W1 / H1 > target_ratio:
            new_w = int(H1 * target_ratio)
            trim = W1 - new_w
            x0 += trim // 2
            x1 -= trim - trim // 2
        else:
            new_h = int(W1 / target_ratio)
            trim = H1 - new_h
            y0 += trim // 2
            y1 -= trim - trim // 2
    return x0, y0, x1, y1


def apply_crop(img, top_frac, right_frac, bottom_frac, left_frac, target_ratio, lock_ratio=True):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = crop_box(w, h, top_frac, right_frac, bottom_frac, left_frac, target_ratio, lock_ratio)
    return img[y0:y1, x0:x1]


def draw_grid(img, step=80, color=(0, 255, 255)):
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), color, 1, cv2.LINE_AA)
    return out


class Calibrator:
    def __init__(self, root, src_path, init_vals=None):
        self.src_path = src_path
        self.full = cv2.imread(src_path)
        if self.full is None:
            raise SystemExit(f"could not read {src_path}")
        h, w = self.full.shape[:2]
        self.target_ratio = w / h
        scale = PREVIEW_MAX_DIM / max(w, h)
        self.preview_src = cv2.resize(self.full, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_AREA)

        self.root = root
        if init_vals is not None:
            # Explicit CLI values always win -- this is the normal way to resume known-good
            # geometry on the ORIGINAL frame (not a pre-corrected one) to tweak tone on top.
            k1_0, k2_0, angle_0, ct_0, cr_0, cb_0, cl_0 = init_vals
            root.title(f"Lens calibrator -- {os.path.basename(src_path)} [resumed from CLI values]")
        else:
            # Re-opening one of this tool's own earlier saves (filename carries "_k1...") means
            # the geometry is already baked into these pixels -- starting the geometry sliders
            # at the usual non-zero guesses would double-apply that correction. Fall back to
            # neutral in that case; otherwise use the usual first-guess starting point.
            already_corrected = "_k1" in os.path.basename(src_path)
            root.title(f"Lens calibrator -- {os.path.basename(src_path)}"
                       + (" [already-corrected input, geometry reset to neutral]" if already_corrected else ""))
            if already_corrected:
                print("Detected an already-corrected input (filename has '_k1...') -- "
                      "starting k1/k2/rotation/crop at neutral (0) instead of the usual guesses.")
            k1_0, k2_0, angle_0 = (0.0, 0.0, 0.0) if already_corrected else (-0.15, 0.04, 3.81)
            ct_0 = cr_0 = cb_0 = cl_0 = 0.0

        self.k1 = tk.DoubleVar(value=k1_0)
        self.k2 = tk.DoubleVar(value=k2_0)
        self.angle = tk.DoubleVar(value=angle_0)
        self.crop_top = tk.DoubleVar(value=ct_0)
        self.crop_right = tk.DoubleVar(value=cr_0)
        self.crop_bottom = tk.DoubleVar(value=cb_0)
        self.crop_left = tk.DoubleVar(value=cl_0)
        self.shadow_lift = tk.DoubleVar(value=0.0)
        self.brightness = tk.DoubleVar(value=0.0)
        self.blacks = tk.DoubleVar(value=0.0)
        self.lock_ratio = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)

        self.img_label = tk.Label(root)
        self.img_label.pack(side=tk.TOP, padx=8, pady=8)

        controls = tk.Frame(root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        self._slider(controls, "k1 (barrel)", self.k1, -1.0, 0.5, 0)
        self._slider(controls, "k2", self.k2, -0.3, 0.3, 1)
        self._slider(controls, "rotation (deg)", self.angle, -20.0, 20.0, 2)
        self._slider(controls, "crop top", self.crop_top, 0.0, 0.4, 3)
        self._slider(controls, "crop right", self.crop_right, 0.0, 0.4, 4)
        self._slider(controls, "crop bottom", self.crop_bottom, 0.0, 0.4, 5)
        self._slider(controls, "crop left", self.crop_left, 0.0, 0.4, 6)
        self._slider(controls, "shadows", self.shadow_lift, 0.0, 1.0, 7)
        self._slider(controls, "brightness", self.brightness, -1.0, 1.0, 8)
        self._slider(controls, "blacks", self.blacks, -1.0, 1.0, 9)

        tk.Checkbutton(controls, text="lock aspect ratio (extra centered trim if needed)",
                        variable=self.lock_ratio, command=self.update_preview
                        ).grid(row=10, column=0, columnspan=3, sticky="w", pady=(6, 0))
        tk.Checkbutton(controls, text="show grid", variable=self.show_grid,
                        command=self.update_preview).grid(row=11, column=0, sticky="w")

        self.status = tk.Label(controls, text="", anchor="w", justify="left", font=("Consolas", 9))
        self.status.grid(row=12, column=0, columnspan=3, sticky="w", pady=(6, 0))

        tk.Button(controls, text="Save full-res PNG + ffmpeg command", command=self.save
                  ).grid(row=13, column=0, columnspan=3, sticky="we", pady=(8, 0))

        self.update_preview()

    def _slider(self, parent, label, var, lo, hi, row):
        tk.Label(parent, text=label, width=14, anchor="w").grid(row=row, column=0, sticky="w")
        s = tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=0.01,
                     orient=tk.HORIZONTAL, length=420, command=lambda _=None: self.update_preview())
        s.grid(row=row, column=1, sticky="we")
        val_label = tk.Label(parent, textvariable=var, width=7)
        val_label.grid(row=row, column=2, sticky="w")

    def update_preview(self):
        src = apply_tone(self.preview_src, self.shadow_lift.get(), self.brightness.get(), self.blacks.get())
        out = apply_transform(src, self.k1.get(), self.k2.get(), self.angle.get())
        out = apply_crop(out, self.crop_top.get(), self.crop_right.get(),
                          self.crop_bottom.get(), self.crop_left.get(),
                          self.target_ratio, self.lock_ratio.get())
        if self.show_grid.get():
            out = draw_grid(out)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.img_label.configure(image=self.tkimg)
        ch, cw = out.shape[:2]
        self.status.configure(
            text=f"k1={self.k1.get():.3f}  k2={self.k2.get():.3f}  rotate={self.angle.get():.2f} deg  "
                 f"crop T/R/B/L={self.crop_top.get():.3f}/{self.crop_right.get():.3f}/"
                 f"{self.crop_bottom.get():.3f}/{self.crop_left.get():.3f}  "
                 f"shadows={self.shadow_lift.get():.2f}  brightness={self.brightness.get():.2f}  "
                 f"blacks={self.blacks.get():.2f}  (preview output {cw}x{ch})"
        )

    def save(self):
        k1, k2, angle = self.k1.get(), self.k2.get(), self.angle.get()
        ct, cr = self.crop_top.get(), self.crop_right.get()
        cb, cl = self.crop_bottom.get(), self.crop_left.get()
        shadow = self.shadow_lift.get()
        brightness = self.brightness.get()
        blacks = self.blacks.get()
        lock = self.lock_ratio.get()
        src_full = apply_tone(self.full, shadow, brightness, blacks)
        out_full = apply_transform(src_full, k1, k2, angle)
        h, w = out_full.shape[:2]
        x0, y0, x1, y1 = crop_box(w, h, ct, cr, cb, cl, self.target_ratio, lock)
        out_full = out_full[y0:y1, x0:x1]
        base, ext = os.path.splitext(self.src_path)
        base = re.sub(r"_k1-?[\d.]+.*$", "", base)  # drop a previous save's own suffix, don't stack it
        out_path = (f"{base}_k1{k1:.3f}_k2{k2:.3f}_rot{angle:.2f}_"
                    f"cropTRBL{ct:.3f}-{cr:.3f}-{cb:.3f}-{cl:.3f}_"
                    f"shadow{shadow:.2f}_bright{brightness:.2f}_blacks{blacks:.2f}{ext}")
        cv2.imwrite(out_path, out_full)
        ffmpeg_filter = (
            f"lenscorrection=cx=0.5:cy=0.5:k1={k1:.3f}:k2={k2:.3f},"
            f"rotate={angle:.3f}*PI/180:fillcolor=black,"
            f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}"
        )
        print(f"saved: {out_path}  ({x1 - x0}x{y1 - y0})")
        print(f"equivalent ffmpeg filter (geometry only, tone adjustments not included): "
              f"{ffmpeg_filter}")
        print(f"for the batch render, pass to apply_lens_correction.py: "
              f"{k1:.3f} {k2:.3f} {angle:.2f} {ct:.3f} {cr:.3f} {cb:.3f} {cl:.3f} "
              f"--shadow {shadow:.2f} --brightness {brightness:.2f} --blacks {blacks:.2f}")
        self.status.configure(
            text=f"saved {os.path.basename(out_path)} ({x1 - x0}x{y1 - y0})\nffmpeg: {ffmpeg_filter}"
        )


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: py lens_calibrator.py <path_to_frame.png> "
                          "[k1 k2 angle_deg crop_top crop_right crop_bottom crop_left]")
    src_path = sys.argv[1]
    init_vals = None
    if len(sys.argv) > 2:
        if len(sys.argv) != 9:
            raise SystemExit("resume form needs all 7 values: "
                              "k1 k2 angle_deg crop_top crop_right crop_bottom crop_left")
        init_vals = tuple(float(a) for a in sys.argv[2:9])
    root = tk.Tk()
    Calibrator(root, src_path, init_vals)
    root.mainloop()


if __name__ == "__main__":
    main()
