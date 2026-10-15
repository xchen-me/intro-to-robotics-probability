"""Demo 3 - Track a coloured object: measurement, prediction, and the growing ellipse.

WHAT TO SET UP
    Hold up something brightly coloured - a tangerine, a red marker cap, a
    sticky note - and move it around in front of the camera. Default threshold
    is tuned for orange/red; use --color green or --color blue otherwise.

    Then, mid-demo, COVER THE OBJECT with your hand for a second or two.

WHAT YOU SEE
    * teal dot  - the raw detection. It jitters frame to frame even when your
      hand is steady: this is measurement noise, exactly like demo 1.
    * purple dot - the Kalman filter estimate. Visibly smoother.
    * purple ellipse - the filter's own 95% position uncertainty.

    While the object is visible the ellipse stays small. The moment you cover
    it, the filter gets no measurement, runs on prediction alone, and the
    ellipse GROWS on screen. Uncover it and the ellipse snaps shut.

OBSERVATIONS
    The filter is not tracking the object. It is tracking its own belief about
    the object, and it tells you how sure it is. Watch the ellipse the moment
    the object is covered - that is prediction inflating uncertainty, live. The
    instant it reappears, one measurement collapses the ellipse again. Predict,
    correct, repeat: that loop is the whole Kalman filter, and you are watching
    it run at 30 Hz.

    The ellipse is drawn at 2.45 sigma, because 95% coverage in two dimensions
    needs 2.45 sigma, not 2.

RUN
    python demo3_tracking_kalman.py --source 0 --color orange
    python demo3_tracking_kalman.py --source demo      # scripted occlusion, no camera
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

PURPLE, GOLD, TEAL, GREY = "#4B2E83", "#B7A57A", "#2E7D9A", "#5F5F6E"
BGR_PURPLE, BGR_TEAL, BGR_GREY = (131, 46, 75), (154, 125, 46), (110, 95, 95)
K95 = 2.4477                       # sqrt(chi2.ppf(0.95, df=2)) - 2D, not 1D

HSV_RANGES = {
    "orange": [((5, 120, 120), (22, 255, 255))],
    "red":    [((0, 120, 110), (8, 255, 255)), ((170, 120, 110), (179, 255, 255))],
    "green":  [((40, 80, 70), (85, 255, 255))],
    "blue":   [((95, 110, 70), (125, 255, 255))],
}


def synthetic_frames(n_frames, size=(640, 480), occlude=(80, 100), seed=3):
    """A moving orange disk on a grey field, hidden for a scripted window."""
    rng = np.random.default_rng(seed)
    w, h = size
    for k in range(n_frames):
        frame = np.full((h, w, 3), 70, np.uint8)
        frame = (frame + rng.normal(0, 3.5, frame.shape)).clip(0, 255).astype(np.uint8)
        t = k / max(n_frames - 1, 1)
        cx = 70 + (w - 140) * t
        cy = h / 2 + 110 * np.sin(2.4 * np.pi * t)
        # Jitter the drawn blob a little: a real detector's centroid wobbles by a
        # couple of pixels even on a steady object, and that wobble is the whole
        # reason a filter helps.
        jx, jy = rng.normal(0, 2.6, 2)
        if not (occlude[0] <= k < occlude[1]):
            cv2.circle(frame, (int(cx + jx), int(cy + jy)), 26, (40, 130, 235), -1)
        yield frame, (cx, cy)


def camera_frames(src, n_frames):
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {src!r}. Try --source demo")
    try:
        for _ in range(15):     # let auto-exposure/white-balance settle before we
            cap.read()          # start - also gives you time to get the object in frame
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, None
    finally:
        cap.release()


def detect(frame, color, min_area=120):
    """Largest blob of the requested colour -> (x, y) centroid, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in HSV_RANGES[color]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, mask
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, mask
    return (M["m10"] / M["m00"], M["m01"] / M["m00"]), mask


def build_filter(meas_sd, accel_sd):
    """Constant-velocity Kalman filter on [x, y, vx, vy]."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                    [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    q = accel_sd ** 2
    kf.processNoiseCov = (q * np.array([[0.25, 0, 0.5, 0], [0, 0.25, 0, 0.5],
                                        [0.5, 0, 1.0, 0], [0, 0.5, 0, 1.0]])).astype(np.float32)
    kf.measurementNoiseCov = (meas_sd ** 2 * np.eye(2)).astype(np.float32)
    kf.errorCovPost = (50.0 * np.eye(4)).astype(np.float32)
    return kf


def ellipse_axes(cov2):
    """(semi-major, semi-minor, angle_deg) of the 95% ellipse of a 2x2 covariance."""
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.clip(vals, 1e-9, None)
    ang = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return K95 * float(np.sqrt(vals[0])), K95 * float(np.sqrt(vals[1])), ang


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="demo")
    ap.add_argument("--color", default="orange", choices=sorted(HSV_RANGES))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--meas-sd", type=float, default=3.0, help="assumed detection sd, px")
    ap.add_argument("--accel-sd", type=float, default=1.2, help="assumed accel sd, px/frame^2")
    ap.add_argument("--save", default="out/demo3_tracking_kalman.png")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    stream = (synthetic_frames(args.frames) if args.source == "demo"
              else camera_frames(args.source, args.frames))
    kf = build_filter(args.meas_sd, args.accel_sd)
    raw, est, semi, gaps, inited, truth = [], [], [], [], [], []
    initialised = False

    for k, (frame, gt) in enumerate(stream):
        z, mask = detect(frame, args.color)
        if z is not None and not initialised:
            kf.statePost = np.array([[z[0]], [z[1]], [0], [0]], np.float32)
            initialised = True
        pred = kf.predict()
        if z is not None and initialised:
            kf.correct(np.array([[z[0]], [z[1]]], np.float32))
        state = kf.statePost if initialised else pred
        a, b, ang = ellipse_axes(np.asarray(kf.errorCovPost)[:2, :2].astype(float))

        raw.append(z if z is not None else (np.nan, np.nan))
        # Before the first detection there is no belief yet to plot - kf.predict()
        # on an unseen filter just returns its [0,0,0,0] default, and plotting that
        # draws a false straight line from the origin to the first real detection.
        est.append((float(state[0, 0]), float(state[1, 0])) if initialised else (np.nan, np.nan))
        semi.append(a if initialised else np.nan)
        gaps.append(z is None)
        inited.append(initialised)
        truth.append(gt if gt is not None else (np.nan, np.nan))

        view = frame.copy()
        for p in est[-40:]:
            if np.isfinite(p[0]):
                cv2.circle(view, (int(p[0]), int(p[1])), 1, BGR_PURPLE, -1)
        if z is not None:
            cv2.circle(view, (int(z[0]), int(z[1])), 5, BGR_TEAL, -1)
        if initialised:
            cv2.ellipse(view, (int(state[0, 0]), int(state[1, 0])),
                        (max(int(a), 1), max(int(b), 1)), ang, 0, 360, BGR_PURPLE, 2)
            cv2.circle(view, (int(state[0, 0]), int(state[1, 0])), 4, BGR_PURPLE, -1)
        if not initialised:
            msg, col = "not yet detected", BGR_GREY
        elif z is None:
            msg, col = f"no measurement - PREDICTING   95% radius {a:5.1f} px", BGR_PURPLE
        else:
            msg, col = f"measurement - correcting   95% radius {a:5.1f} px", BGR_TEAL
        cv2.putText(view, msg, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if show_frame(view, args.headless):
            break
    if not args.headless:
        cv2.destroyAllWindows()

    raw, est, semi = np.asarray(raw, float), np.asarray(est, float), np.asarray(semi, float)
    gaps, inited = np.asarray(gaps), np.asarray(inited)
    vis = ~gaps
    # "Occluded" means a measurement went missing AFTER the filter already had a
    # belief to coast on. Frames before the first detection are neither - exclude
    # them so the pre-lock-on gap doesn't get mistaken for a covered object.
    occluded = gaps & inited
    print(f"frames processed       {len(semi)}")
    print(f"frames before first detection  {int((~inited).sum())}")
    print(f"frames with detection  {int(vis.sum())}   without (after lock-on): {int(occluded.sum())}")
    if vis.any():
        print(f"95% radius, tracking   {np.median(semi[vis]):.1f} px (median)")
    if occluded.any():
        print(f"95% radius, occluded   {semi[occluded].max():.1f} px (peak)")
        print(f"growth factor          {semi[occluded].max() / max(np.median(semi[vis]), 1e-9):.1f}x")
    if np.isfinite(raw).any():
        jitter = np.nanstd(np.diff(raw[:, 0])) / np.sqrt(2)
        print(f"detection jitter       {jitter:.2f} px (sd of x, from successive differences)")
    print(f"assumed measurement sd {args.meas_sd:.1f} px")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.9),
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.88, wspace=0.26)
    a1.plot(raw[:, 0], raw[:, 1], "o", ms=3.2, color=TEAL, alpha=0.65, label="raw detection")
    a1.plot(est[:, 0], est[:, 1], "-", lw=2.2, color=PURPLE, label="filter estimate")
    if occluded.any():
        a1.plot(est[occluded, 0], est[occluded, 1], "-", lw=3.0, color=GOLD, label="no measurement")
    a1.invert_yaxis(); a1.set_xlabel("x (px)"); a1.set_ylabel("y (px)")
    a1.set_title("Raw detection vs filtered estimate")
    a1.legend(frameon=False, fontsize=9)

    semi_max = np.nanmax(semi) if np.isfinite(semi).any() else 1.0
    a2.plot(semi, lw=2.2, color=PURPLE)
    if occluded.any():
        a2.fill_between(np.arange(len(semi)), 0, semi_max * 1.1, where=occluded,
                        color=GOLD, alpha=0.30, lw=0)
        mid = int(np.mean(np.flatnonzero(occluded)))
        a2.text(mid, semi_max * 1.02, "occluded", color=GREY, ha="center",
                va="top", fontsize=9)
    a2.set_ylim(0, semi_max * 1.12)
    a2.set_xlabel("frame"); a2.set_ylabel("95% position radius (px)")
    a2.set_title("Uncertainty inflates, then collapses")
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=150)
    print(f"figure                 {args.save}")
    if not args.headless:
        plt.show()


def show_frame(image, headless):
    if headless:
        return False
    try:
        cv2.imshow("demo 3 - track and predict", image)
        return (cv2.waitKey(1) & 0xFF) in (27, ord("q"))
    except cv2.error:
        return False


if __name__ == "__main__":
    main()
