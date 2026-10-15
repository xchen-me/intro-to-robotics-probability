"""Demo 2 - Averaging frames: where 1/sqrt(n) comes from, and where it stops.

WHAT TO SET UP
    Same as demo 1: camera still, scene static. We now average frames together.

WHAT YOU SEE
    Left: a single frame crop next to the average of all N frames. Watch the
    averaged image come out visibly cleaner - the noise is disappearing.

    Right: the measured noise standard deviation against n, with the
    sigma/sqrt(n) prediction drawn through it. They agree across two decades.

    The measurement differences two INDEPENDENT n-frame averages, so the scene
    cancels and no knowledge of the "true" image is needed - that is the only
    honest way to measure noise from a camera you cannot calibrate.

OBSERVATIONS
    Averaging n independent readings divides the random error by root n. Four
    frames, half the noise. A hundred frames, a tenth. That is the single most
    useful fact about estimation.

    And the honest half: this plot only tracks the RANDOM part, because
    differencing two averages cancels anything repeatable. That is exactly the
    point - whatever is repeatable is also what averaging cannot touch. A
    per-pixel offset, a dead pixel, a mis-calibrated sensor: every one of those
    survives all 256 frames. The script prints how much offset is still there
    after averaging. You cannot average your way out of a systematic error.

RUN
    python demo2_frame_averaging.py --source 0 --frames 256
    python demo2_frame_averaging.py --source demo
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
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--patch", type=int, default=120, help="side of the measurement patch")
    ap.add_argument("--save", default="out/demo2_frame_averaging.png")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    stack = []
    for frame in frame_source(args.source, args.frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        stack.append(gray)
        if len(stack) % 8 == 0:
            avg = np.mean(stack, axis=0)
            side = np.hstack([stack[0], avg]).astype(np.uint8)
            cv2.putText(side, f"single frame | average of {len(stack)}", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            if show("demo 2 - averaging", side, args.headless):
                break
    cv2.destroyAllWindows() if not args.headless else None

    S = np.asarray(stack)                       # (n, h, w)
    n_total, h, w = S.shape
    s = args.patch
    y0, x0 = (h - s) // 2, (w - s) // 2
    patch = S[:, y0:y0 + s, x0:x0 + s]

    # How do you measure noise without knowing the true scene? Difference two
    # INDEPENDENT n-frame averages: the scene cancels, and so does any fixed
    # per-pixel bias, leaving only the random part. Dividing by sqrt(2) undoes
    # the variance doubling of a difference.
    #
    # Comparing against the full-stack average instead would be self-referential
    # - at n = n_total the answer is exactly zero by construction - which is why
    # that tempting shortcut is not used here.
    half = n_total // 2
    a_half, b_half = patch[:half], patch[half:2 * half]
    ns = np.unique(np.round(np.logspace(0, np.log10(half), 12)).astype(int))
    sds = [float((a_half[:k].mean(axis=0) - b_half[:k].mean(axis=0)).std() / np.sqrt(2))
           for k in ns]
    sd1 = sds[0]

    print(f"frames captured        {n_total}  (split into two halves of {half})")
    print(f"patch                  {s}x{s} px at ({x0},{y0})")
    print(f"sd of a single frame   {sd1:.3f} DN")
    for k in (1, 4, 16, 64):
        if k <= half:
            i = int(np.argmin(np.abs(ns - k)))
            print(f"  n = {ns[i]:<4d} measured {sds[i]:.3f}   "
                  f"1/sqrt(n) predicts {sd1 / np.sqrt(ns[i]):.3f}   "
                  f"ratio {sds[i] / (sd1 / np.sqrt(ns[i])):.3f}")
    print(f"largest deviation      "
          f"{max(abs(m / (sd1 / np.sqrt(k)) - 1) for k, m in zip(ns, sds)) * 100:.1f}% "
          f"from the 1/sqrt(n) line")
    fpn = float((patch.mean(axis=0) - cv2.blur(patch.mean(axis=0), (25, 25))).std())
    print(f"structure left after   {fpn:.3f} DN of per-pixel offset survives averaging")
    print(f"  averaging {n_total} frames  (the difference test above cancels it, and so "
          f"does nothing else)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9),
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.subplots_adjust(left=0.02, right=0.975, bottom=0.16, top=0.88, wspace=0.18)
    full_avg = patch.mean(axis=0)
    crop = np.hstack([S[0, y0:y0 + s, x0:x0 + s], full_avg])
    a1.imshow(crop, cmap="gray"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title(f"one frame  |  average of {n_total}")
    a1.axvline(s, color="white", lw=1.5)

    a2.plot(ns, sds, "o-", color=TEAL, ms=5, lw=1.6, label="measured")
    a2.plot(ns, sd1 / np.sqrt(ns), color=PURPLE, lw=2.2, ls="--",
            label=r"$\sigma/\sqrt{n}$")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("frames averaged, $n$"); a2.set_ylabel("noise sd (DN)")
    a2.set_title(r"Noise falls as $1/\sqrt{n}$")
    a2.legend(frameon=False, fontsize=9, loc="lower left")
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=150)
    print(f"figure                 {args.save}")
    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
