# Intro to Robotics: Probability — Demos and Companion Code

This repo has two things from the *Foundations of Probability and Estimation*
lecture, and neither needs the lecture slides to run:

- **Camera demos** (`demo1`–`demo4`, root of this repo) — four live demos that
  make core probability and estimation ideas visible on a webcam feed: pixel
  noise, averaging, Kalman tracking, and per-pixel background modeling. Every
  demo also runs with **no camera attached** (`--source demo`), so you can
  follow along on a laptop with no webcam.
- **Lecture 1 companion code** ([`l1-lecture-code/`](l1-lecture-code)) —
  standalone scripts behind *Foundations of Probability*: summing uniform
  random variables to watch the Central Limit Theorem kick in, the 30 cm
  wall-sensor demo with an adjustable noise level, and the CLT on a bimodal
  mixture. Same idea — no slides needed, just Python (or MATLAB for one of
  them).

Pick whichever one you're here for. They don't depend on each other, and the
Python setup below (Section 1) covers both.

## What's here

**Camera demos** — need a webcam, or run with `--source demo`:

| Script | What it shows |
|---|---|
| `demo1_pixel_noise.py` | A single pixel's brightness, read 300 times from a scene that never changes — a measurement is a distribution. |
| `demo2_frame_averaging.py` | Averaging 256 frames: noise falls as $1/\sqrt n$, and what averaging *cannot* fix. |
| `demo3_tracking_kalman.py` | A Kalman filter tracking a colored object, with its own uncertainty ellipse growing when you cover the object and collapsing when you uncover it. |
| `demo4_background_gaussian.py` | Learning a per-pixel mean/standard-deviation model of an empty scene, then flagging anything that doesn't fit it. |

**Lecture 1 companion code** ([`l1-lecture-code/`](l1-lecture-code)) — no camera, just numbers:

| Script | What it shows |
|---|---|
| `clt_uniform.py` (+ `clt_uniform_demo.m` for MATLAB) | Summing $n = 1,2,3,12$ uniform random variables: raw sums broaden, centered/scaled sums approach $N(0,1)$ — the Central Limit Theorem. |
| `range_measurement_demo.py --sigma 0.02` | A 30 cm wall-distance sensor with adjustable noise, the running example for "measurement = truth + noise." |
| `summed_disturbances_demo.py` | The CLT showing up again on a bimodal mixture, not just a uniform distribution. |

See [`l1-lecture-code/README.txt`](l1-lecture-code/README.txt) for exact run commands and expected output; Section 1 below gets Python ready for both halves of the repo.

## 1. Set up Python

You need Python 3.9 or newer either way. Pick whichever of these matches you —
they produce the same working environment, just by two different routes.

### Option A — You already did the `StudentLabs` setup

If you've gone through this course's lab setup (the lab laptop's Ubuntu, or
WSL2 Ubuntu-24.04 on your own Windows computer, set up for ROS 2 Jazzy per
the course's lab instructions), you already have a system Python 3 with
`numpy` and `matplotlib` installed —
ROS 2 Jazzy on Ubuntu 24.04 pulls those in as `python3-numpy` and
`python3-matplotlib`. That environment works for this repo as-is. Don't put
it in a venv — ROS workflows (`colcon build`, `source install/setup.bash`)
expect the system interpreter, and a venv can break that.

In the same Ubuntu terminal you use for labs (native, or the WSL2 one):

```bash
sudo apt install -y python3-opencv   # the one package the labs don't already install
git clone https://github.com/xchen-me/intro-to-robotics-probability.git
cd intro-to-robotics-probability
python3 demo1_pixel_noise.py --source demo   # smoke test, see Section 2
```

If that smoke test complains about a missing or too-old package instead of
popping up a plot, your ROS 2 install didn't pull in everything this repo's
`requirements.txt` lists — `pip install -r requirements.txt --user` fills in
whatever's missing without disturbing the rest of your ROS setup.

**On macOS or native Linux instead of WSL2?** You still have Python 3 from
the labs' setup, just without ROS's apt packages. Jump to Option B below —
the "already have Python 3.9+" branch of it is exactly this case, just
`pip install -r requirements.txt` with no `venv` required if you'd rather not
bother with one for a short demo session.

### Option B — Your own computer, from scratch (pyenv)

If you're not working from a lab-configured machine, use
[`pyenv`](https://github.com/pyenv/pyenv) to get a clean, known-good Python
without touching your system's default one:

```bash
# macOS
brew install pyenv
# Linux
curl -fsSL https://pyenv.run | bash
# (then add pyenv to your shell init as its installer prints, and restart your terminal)

pyenv install 3.12.6
git clone https://github.com/xchen-me/intro-to-robotics-probability.git
cd intro-to-robotics-probability
pyenv local 3.12.6          # pins this folder to 3.12.6, writes .python-version
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Every command in the rest of this README assumes that venv is still
activated (`source .venv/bin/activate` again in any new terminal). Using
`l1-lecture-code/`? It has its own, shorter `requirements.txt` — see its
[README.txt](l1-lecture-code/README.txt) — install from inside the same venv.

No `pyenv`, just a plain `python3` you're happy with? Skip straight to
`python3 -m venv .venv` above; `pyenv` only matters if you want a specific,
reproducible Python version rather than whatever your OS ships.

## 2. Smoke-test with no camera

Every camera-demo script can synthesize its own data, so you can confirm your
install works before touching a webcam:

```bash
python3 demo1_pixel_noise.py --source demo
```

A window should pop up with a plot in it. (If you add `--headless`, there's
no window — instead you'll see a `figure  out/demo1_pixel_noise.png` line
printed, and the PNG will appear in `out/`.) Either way, if that runs with no
errors, you're set up correctly.

## 3. Find your camera, then run each demo

Before guessing `--source 0`, see what cameras OpenCV can actually reach and
at what index — this matters most on a laptop with more than one camera
(built-in FaceTime/webcam plus an external one), where index `0` is not
always the one you expect:

```bash
python3 demo1_pixel_noise.py --list-cameras
```

This opens every camera index it can find and reports its resolution (and,
on macOS, the camera names from System Settings for reference). Pick the
`--source` number whose resolution matches the camera you actually want —
for example, the built-in FaceTime camera is usually a smaller/different
resolution than a plugged-in external one, and it is **not** always index
`0`.

```bash
python3 demo1_pixel_noise.py --source 0
python3 demo2_frame_averaging.py --source 0 --frames 256
python3 demo3_tracking_kalman.py --source 0 --color orange
python3 demo4_background_gaussian.py --source 0 --train 120
```

Prefer not to type four commands? `make` runs all four on the synthetic
source and refreshes every figure in `out/`; `make SOURCE=0` does the same
against your camera (after you've confirmed `0` is the right one with
`--list-cameras`).

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

## Flags every camera demo accepts

| Flag | Meaning |
|---|---|
| `--source 0` | camera index (run `--list-cameras` first if you're not sure which number is which camera), a video file, or `demo` |
| `--list-cameras` | print every camera index OpenCV can open, with its resolution, then exit — doesn't run the demo |
| `--max-index N` | how many indices `--list-cameras` probes (default 6) — raise it if your camera sits higher |
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

**`could not open source '0'`.** Wrong camera index — run `--list-cameras`
to see which indices actually open a camera on your machine.

**It keeps opening the wrong camera (e.g. an external webcam, when I want
the built-in FaceTime/laptop camera).** This is the most common surprise on
a laptop with two cameras: OpenCV's index order does not always match what
you'd guess, and the external one is often `0`. Run `--list-cameras` — it
prints the resolution at each index (and the macOS camera-name list, for
reference) — and then pass `--source` with whichever index gave the
resolution of the camera you actually want, e.g. `--source 1`.

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
