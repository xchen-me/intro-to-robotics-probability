"""Demo 1 - Your camera is the noisy sensor. One pixel, many frames.

WHAT TO SET UP
    Point the webcam at anything that does not move - a wall, a book, the
    ceiling. Hold the camera still (or tape it down). Nothing in the scene
    changes for the whole recording.

WHAT YOU SEE
    We watch ONE pixel for 300 frames and plot the number it reports. The scene
    is constant, the pixel is constant, and the value is different every frame.
    Pooled into a histogram, those values have a clean bell shape with a
    measurable standard deviation in digital numbers (DN).

OBSERVATIONS
    This is a 30-dollar sensor telling you the brightness of a wall that is not
    changing, and it never reports the same number twice. The mean is the
    brightness; the spread is the honesty. Any sensor you will ever use behaves
    this way - your camera just lets you watch it happen in real time.

    The punchline for the course: a measurement is not a number, it is a
    distribution, and the width of that distribution is a specification.

RUN
    python demo1_pixel_noise.py --list-cameras      # which --source is which camera?
    python demo1_pixel_noise.py --source 0          # your webcam
    python demo1_pixel_noise.py --source demo       # no camera needed
    python demo1_pixel_noise.py --source 0 --pixel 320 240 --frames 500
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
if "--headless" in sys.argv:
    matplotlib.use("Agg")           # no display to pop a window on
import matplotlib.pyplot as plt

from camera_utils import print_camera_list

# ---------------------------------------------------------------- palette --
PURPLE, GOLD, TEAL, GREY = "#4B2E83", "#B7A57A", "#2E7D9A", "#5F5F6E"
BGR_PURPLE, BGR_GOLD, BGR_TEAL, BGR_GREY = (131, 46, 75), (122, 165, 183), (154, 125, 46), (110, 95, 95)


def frame_source(src, n_frames, size=(640, 480), seed=7):
    """Yield BGR frames from a camera index, a video file, or 'demo'.

    'demo' synthesises a static scene with realistic sensor noise so every demo
    runs with no camera attached - the analysis code downstream is identical.
    """
    if src != "demo":
        cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        if not cap.isOpened():
            raise SystemExit(f"could not open source {src!r}. Run --list-cameras to see "
                              "which indices open a camera, or try --source demo")
        try:
            for _ in range(15):     # let auto-exposure/white-balance settle before
                cap.read()          # we start recording - the first live frames off
                                     # a freshly opened camera are often under-exposed
            for _ in range(n_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
        return

    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    scene = (90 + 70 * (xx / w) + 40 * np.sin(yy / 55.0)).astype(np.float32)
    scene = np.dstack([scene * 0.92, scene, scene * 1.05])          # mild colour cast
    # Fixed-pattern noise is a property of the SENSOR, not of the recording, so
    # it gets its own fixed seed: every call to frame_source sees the same
    # per-pixel offsets, exactly as a real camera would behave.
    fixed_pattern = np.random.default_rng(1234).normal(0, 1.4, scene.shape).astype(np.float32)
    for _ in range(n_frames):
        read_noise = rng.normal(0, 2.6, scene.shape)
        shot = rng.normal(0, 1.0, scene.shape) * np.sqrt(np.clip(scene, 1, None)) * 0.16
        yield np.clip(scene + fixed_pattern + read_noise + shot, 0, 255).astype(np.uint8)


def show(window, image, headless):
    """imshow that degrades to a no-op when there is no display."""
    if headless:
        return False
    try:
        cv2.imshow(window, image)
        return (cv2.waitKey(1) & 0xFF) in (27, ord("q"))
    except cv2.error:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="demo", help="camera index, video file, or 'demo'")
    ap.add_argument("--list-cameras", action="store_true",
                     help="print which --source index opens which camera, then exit")
    ap.add_argument("--max-index", type=int, default=6,
                     help="highest camera index to probe for --list-cameras")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--pixel", type=int, nargs=2, default=None, help="x y (default: centre)")
    ap.add_argument("--save", default="out/demo1_pixel_noise.png")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    if args.list_cameras:
        print_camera_list(args.max_index)
        return

    values, px, py = [], None, None
    for frame in frame_source(args.source, args.frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if px is None:
            h, w = gray.shape
            px, py = args.pixel if args.pixel else (w // 2, h // 2)
        values.append(float(gray[py, px]))

        view = frame.copy()
        cv2.drawMarker(view, (px, py), BGR_PURPLE, cv2.MARKER_CROSS, 26, 2)
        cv2.rectangle(view, (px - 14, py - 14), (px + 14, py + 14), BGR_PURPLE, 2)
        cv2.putText(view, f"pixel ({px},{py}) = {values[-1]:.0f} DN   frame {len(values)}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR_PURPLE, 2)
        if show("demo 1 - one pixel", view, args.headless):
            break
    cv2.destroyAllWindows() if not args.headless else None

    v = np.asarray(values)
    mu, sd = v.mean(), v.std(ddof=1)
    within = float(np.mean(np.abs(v - mu) <= sd))
    print(f"frames captured      {v.size}")
    print(f"pixel               ({px}, {py})")
    print(f"mean                 {mu:.2f} DN")
    print(f"standard deviation   {sd:.2f} DN")
    print(f"range observed       {v.min():.0f} to {v.max():.0f} DN")
    print(f"fraction within 1 sd {within:.3f}   (Gaussian says 0.683)")
    print(f"distinct values seen {len(np.unique(v))}")
    if len(np.unique(v)) < 20:
        print("  note: the sensor quantises to integer DN, so with a sd of only a "
              "few DN the\n        empirical 68% figure can sit a few points off the "
              "Gaussian value - that is\n        discreteness, not a broken sensor. "
              "Worth saying out loud if a student asks.")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.6),
                                 gridspec_kw=dict(width_ratios=[1.45, 1]))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.88, wspace=0.24)
    k = min(160, v.size)
    a1.axhline(mu, color=GREY, lw=1.2, ls="--")
    a1.plot(np.arange(k), v[:k], marker="o", ms=2.8, lw=0.7, color=TEAL)
    a1.set_xlabel("frame"); a1.set_ylabel("pixel value (DN)")
    a1.set_title("Nothing moved; the number changed anyway")
    a1.set_xlim(-2, k * 1.18)
    a1.text(k * 1.16, mu, f"mean\n{mu:.1f}", color=GREY, ha="right", va="center", fontsize=9)

    lo, hi = mu - 4.2 * sd, mu + 4.2 * sd
    a2.hist(v, bins=np.arange(np.floor(lo), np.ceil(hi) + 1) - 0.5, density=True,
            color=GOLD, alpha=0.55, edgecolor=GOLD)
    xs = np.linspace(lo, hi, 300)
    a2.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
            color=PURPLE, lw=2.4)
    a2.set_xlabel("pixel value (DN)"); a2.set_ylabel("density (1/DN)")
    a2.set_title("Pooled, it has a shape")
    a2.text(0.97, 0.92, f"$\\sigma$ = {sd:.2f} DN", transform=a2.transAxes,
            ha="right", va="top", color=PURPLE, fontsize=11)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=150)
    print(f"figure               {args.save}")
    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
