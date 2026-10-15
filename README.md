# Intro to Robotics: Probability — Camera Demos

Four live demos that make core probability and estimation ideas visible on a
webcam feed: pixel noise, averaging, Kalman tracking, and per-pixel
background modeling. Every demo also runs with **no camera attached**
(`--source demo`), so you can follow along on a laptop with no webcam.

## What's here

| Script | What it shows |
|---|---|
| `demo1_pixel_noise.py` | A single pixel's brightness, read 300 times from a scene that never changes — a measurement is a distribution. |
| `demo2_frame_averaging.py` | Averaging 256 frames: noise falls as $1/\sqrt n$, and what averaging *cannot* fix. |
| `demo3_tracking_kalman.py` | A Kalman filter tracking a colored object, with its own uncertainty ellipse growing when you cover the object and collapsing when you uncover it. |
| `demo4_background_gaussian.py` | Learning a per-pixel mean/standard-deviation model of an empty scene, then flagging anything that doesn't fit it. |

## 1. Install Python and the dependencies

You need Python 3.9 or newer. Check what you have:

```bash
python3 --version
```

Then install the three packages every demo needs:

```bash
pip install -r requirements.txt
```

or, with conda:

```bash
conda create -n robotics-probability -c conda-forge python=3.12 py-opencv numpy matplotlib
conda activate robotics-probability
```

## 2. Smoke-test with no camera

Every script can synthesize its own data, so you can confirm your install
works before touching a webcam:

```bash
python3 demo1_pixel_noise.py --source demo
```

A window should pop up with a plot in it. (If you add `--headless`, there's
no window — instead you'll see a `figure  out/demo1_pixel_noise.png` line
printed, and the PNG will appear in `out/`.) Either way, if that runs with no
errors, you're set up correctly.

## 3. Run each demo on your own camera

```bash
python3 demo1_pixel_noise.py --source 0
python3 demo2_frame_averaging.py --source 0 --frames 256
python3 demo3_tracking_kalman.py --source 0 --color orange
python3 demo4_background_gaussian.py --source 0 --train 120
```

`--source 0` is your default webcam. If that opens the wrong camera (common
on a laptop with both a built-in and an external one), try `--source 1`.

Prefer not to type four commands? `make` runs all four on the synthetic
source and refreshes every figure in `out/`; `make SOURCE=0` does the same
against your camera.

What to actually do in front of each camera:

1. **Demo 1** — point the camera at a wall or anything that does not move.
   Keep the camera still for the whole 300-frame recording.
2. **Demo 2** — same setup as demo 1: camera still, scene static.
3. **Demo 3** — hold up something brightly colored (a tangerine, a red
   marker cap). Move it around, then **cover it with your hand** for a
   couple of seconds, then uncover it. The default color threshold is tuned
   for orange/red — pass `--color green` or `--color blue` for something
   else.
4. **Demo 4** — keep the scene empty for the first ~120 frames (that's the
   training phase), then walk into frame or wave a hand through it.

Every run finishes two ways: it prints the numbers it measured straight to
your terminal, and it pops the summary figure up in its own window (also
saved to `out/`, so you can come back to it later). Press `q` or `Esc` to
stop a live window early.

## Flags every demo accepts

| Flag | Meaning |
|---|---|
| `--source 0` | camera index (try `1` if `0` opens the wrong camera), a video file, or `demo` |
| `--frames N` | how many frames to capture |
| `--headless` | skip every window — live capture and the pop-up figure — useful if you're running this somewhere with no display |
| `--save PATH` | where the summary figure gets saved (default `out/<demo-name>.png`) |

## Why does the demo use `--source demo` sometimes?

Every script can synthesize a plausible static scene with realistic sensor
noise, entirely in software — no camera involved. That is `--source demo`,
the default if you don't pass `--source` at all. It exists so the *analysis*
part of each script (the math, the plots, the printed numbers) runs through
exactly the same code path whether or not a camera is attached, so you can
explore every demo on a laptop with no webcam.

## Troubleshooting

**macOS asks for camera permission, or the window is black.** Give your
terminal app Camera access in System Settings → Privacy & Security → Camera,
then restart the terminal.

**`could not open source '0'`.** Wrong camera index — try `--source 1`.

**Demo 3 doesn't see my object.** Color thresholds are lighting-dependent.
Try `--color red` / `green` / `blue`, improve the lighting, or use a more
saturated object. A matte orange item on a plain background works best. If
you want to tune it yourself, the `HSV_RANGES` dictionary near the top of
`demo3_tracking_kalman.py` is two numbers per color to edit.

**Demo 3 keeps locking onto my face with `--color red`.** Skin — especially
the nose, cheeks and lips under warm indoor light — sits in the same
low-hue band as "red," so this is a real limitation of plain HSV
thresholding, not a bug. `red`'s saturation/value floor is already set
higher than the other colors to reduce it, but if it still happens: keep
your face out of frame, hold the object further away, or just use
`--color orange` / `green` / `blue` instead.

**Demo 3's ellipse looks permanently huge, or permanently tiny.**
`--meas-sd` tells the filter how noisy your detector is; `--accel-sd` tells
it how fast the object can change velocity. Raise `--accel-sd` if the
filter lags behind fast motion; raise `--meas-sd` if the estimate chases
every jitter.

**Demo 4's mask comes out all white.** The scene changed between training
and test — this usually means your camera's auto-exposure or
auto-white-balance re-adjusted mid-run, which counts as "the scene changed"
from the model's point of view. Lock exposure if your camera driver allows
it, or just re-run the training phase.

**The single frame in demo 2 looks much darker than the average.** The very
first frame a camera returns right after it opens is sometimes
under-exposed while auto-exposure is still converging. Every script already
discards the first 15 frames to let this settle; if you still see it, give
the camera another second before you start recording, or just rerun the
demo.

**It runs but no window appears.** You likely have a headless build of
OpenCV installed. `pip install opencv-python` (not
`opencv-python-headless`), or just pass `--headless` and look at the saved
figure in `out/` instead.

## Regenerating the setup illustrations (optional)

`make_illustrations.py` draws the small conceptual diagrams in `assets/`
(camera + wall, independent-noise-vs-fixed-offset, the predict/correct loop,
a camera tracking a moving object, per-pixel Gaussians, and a QR code
linking back to this repo) with matplotlib. You don't need to run it to use
the demos — it's here for anyone extending this material.

```bash
python3 make_illustrations.py
```

The QR code needs one extra package not in `requirements.txt`
(`pip install "qrcode[pil]"`); if it's missing, the script just skips that
one image and prints a note instead of failing.

## Also in this repo: Lecture 1 companion code

[`l1-lecture-code/`](l1-lecture-code) has the standalone scripts behind
*Foundations of Probability* — summing uniform random variables to watch
the Central Limit Theorem kick in (`clt_uniform.py`, plus a MATLAB
equivalent), the 30 cm wall-sensor demo with an adjustable noise level
(`range_measurement_demo.py --sigma 0.02`), and the CLT on a bimodal
mixture (`summed_disturbances_demo.py`). See that folder's own `README.txt`
for setup and usage.
