"""Demo 4 - Every pixel has its own Gaussian: background modelling as probability.

WHAT TO SET UP
    Camera still, scene empty, for the first few seconds - that is the training
    phase. Then walk into frame, or wave a hand through it.

WHAT YOU SEE
    Three images built from the training frames:
      * the MEAN image - what the camera expects to see, per pixel;
      * the SD image - how much each pixel normally wobbles. It is not uniform:
        edges and bright regions are noisier, flat dark areas are quieter;
      * the z-score map of a new frame, (value - mean) / sd, thresholded at 3.

    The thresholded map is a foreground mask - and it was built from nothing but
    a mean and a standard deviation per pixel.

OBSERVATIONS
    This is what cv2.createBackgroundSubtractorMOG2 is doing under the hood,
    and it is pure probability: each pixel carries a Gaussian, and "foreground"
    means "this reading is too unlikely under that pixel's own distribution."

    Notice the threshold is not picked in pixel values - it is picked in
    SIGMAS. Three sigma means accepting a 0.27% false-alarm rate per pixel. On
    a 640x480 frame that is about 800 pixels of noise every frame, which is
    exactly why the mask then gets eroded and dilated. The threshold is not a
    magic number; it is a false-alarm budget.

RUN
    python demo4_background_gaussian.py --source 0 --train 120
    python demo4_background_gaussian.py --source demo
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
if "--headless" in sys.argv:
    matplotlib.use("Agg")           # no display to pop a window on
import matplotlib.pyplot as plt

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
            raise SystemExit(f"could not open source {src!r}. Try --source demo")
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
    ap.add_argument("--source", default="demo")
    ap.add_argument("--train", type=int, default=120, help="frames used to learn the model")
    ap.add_argument("--z", type=float, default=3.0, help="threshold in sigmas")
    ap.add_argument("--save", default="out/demo4_background_gaussian.png")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    # ---- training phase: learn mean and sd per pixel ----------------------
    stack = []
    for frame in frame_source(args.source, args.train):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        stack.append(gray)
        view = gray.astype(np.uint8).copy()
        cv2.putText(view, f"training {len(stack)}/{args.train} - keep the scene empty",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        if show("demo 4 - background model", view, args.headless):
            break
    S = np.asarray(stack)
    mean, sd = S.mean(axis=0), S.std(axis=0, ddof=1)
    sd = np.maximum(sd, 0.6)                      # floor: a dead-flat pixel is not certain

    # ---- a background-only frame: how many false alarms do we get? --------
    bg_only = next(iter(frame_source(args.source, 1, seed=99)))
    bg_gray = cv2.cvtColor(bg_only, cv2.COLOR_BGR2GRAY).astype(np.float32)
    z_bg = (bg_gray - mean) / sd
    false_frac = float(np.mean(np.abs(z_bg) > args.z))

    # ---- a frame with something in it -------------------------------------
    test = bg_only.copy()
    if args.source == "demo":
        h, w = bg_gray.shape
        cv2.circle(test, (int(w * 0.62), int(h * 0.55)), 54, (210, 210, 210), -1)
        cv2.rectangle(test, (int(w * 0.18), int(h * 0.30)),
                      (int(w * 0.30), int(h * 0.78)), (35, 35, 35), -1)
    else:
        for frame in frame_source(args.source, 1):
            test = frame
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32)
    z = (test_gray - mean) / sd
    mask = (np.abs(z) > args.z).astype(np.uint8) * 255
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n_px = mask.size
    print(f"training frames        {len(stack)}")
    print(f"per-pixel sd           median {np.median(sd):.2f} DN, "
          f"range {sd.min():.2f} to {sd.max():.2f}")
    print(f"threshold              |z| > {args.z:.1f}")
    tail = 2 * (1 - 0.5 * (1 + math.erf(args.z / math.sqrt(2))))
    print(f"expected false alarms  {tail:.5f} of pixels  ({n_px * tail:.0f} px)")
    print(f"observed on empty bg   {false_frac:.5f} of pixels  "
          f"({int(false_frac * n_px)} px)")
    print(f"flagged on test frame  {int((mask > 0).sum())} px  "
          f"-> {int((cleaned > 0).sum())} px after opening")

    fig, axs = plt.subplots(1, 4, figsize=(13.2, 3.1))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.88, wspace=0.08)
    axs[0].imshow(mean, cmap="gray"); axs[0].set_title("mean: what the camera expects")
    im = axs[1].imshow(sd, cmap="magma"); axs[1].set_title("sd: how much each pixel wobbles")
    fig.colorbar(im, ax=axs[1], fraction=0.045, label="DN")
    axs[2].imshow(np.clip(np.abs(z), 0, 8), cmap="viridis")
    axs[2].set_title(r"$|z| = |x-\mu|/\sigma$ on a new frame")
    axs[3].imshow(cleaned, cmap="gray")
    axs[3].set_title(f"$|z| > {args.z:.0f}$, then opened")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=150)
    print(f"figure                 {args.save}")
    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
