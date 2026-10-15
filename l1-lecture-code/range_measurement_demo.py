"""Lecture 1: a stationary range sensor, histogrammed down to a density.

Author: Xu Chen

Setup: a proximity sensor faces a flat wall at a true standoff of 30 cm and
never moves. We log 30,000 independent readings, X ~ N(0.30, sigma^2) in
metres.
Assumptions: independent readings, a fixed Gaussian model, zero bias.
Takeaway: shrinking the histogram bin width does not change what a bar
represents - its AREA is always the fraction of readings in that bin. As
bins narrow, the bar tops converge onto the density p_X(x), and
Pr(0.28 < X <= 0.32) is the shaded AREA under that curve, not a bar height.

Run from any directory: python path/to/range_measurement_demo.py [--sigma 0.02] [--no-show]
Requires NumPy and Matplotlib. Saves a plot and a JSON summary to
modernized_lectures/code-results/l1.
"""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRUE_STANDOFF = 0.30  # metres


def simulate(sigma, trials=30_000, seed=2026):
    rng = np.random.default_rng(seed)
    return rng.normal(TRUE_STANDOFF, sigma, size=trials)


def run(sigma=0.02, trials=30_000, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = ROOT / "code-results/l1"
    output.mkdir(parents=True, exist_ok=True)

    readings = simulate(sigma, trials)
    lo, hi = TRUE_STANDOFF - sigma, TRUE_STANDOFF + sigma
    within_1sd = float(np.mean((readings > lo) & (readings <= hi)))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    edges = np.linspace(TRUE_STANDOFF - 4 * sigma, TRUE_STANDOFF + 4 * sigma, 81)
    heights, edges = np.histogram(readings, bins=edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.stairs(heights, edges, fill=True, alpha=0.55, color="#4B2E83",
              label="measured density")

    grid = np.linspace(edges[0], edges[-1], 400)
    phi = np.exp(-0.5 * ((grid - TRUE_STANDOFF) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(grid, phi, color="#9B761F", lw=2, label=r"$N(0.30,\sigma^2)$ model")

    shade = (grid > lo) & (grid <= hi)
    ax.fill_between(grid[shade], 0, phi[shade], color="#9B761F", alpha=0.25)

    ax.set_xlabel("range (m)")
    ax.set_ylabel(r"$p_X(x)$  (1/m)")
    ax.set_title(f"sigma = {sigma*100:.1f} cm")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "range_measurement.png", dpi=180)
    fig.savefig(output / "range_measurement.pdf")

    summary = dict(sigma=sigma, trials=trials, mean=float(readings.mean()),
                   std=float(readings.std(ddof=1)),
                   fraction_within_1sd=within_1sd,
                   gaussian_prediction=0.6827)
    (output / "range_measurement_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"sigma (assumed)      {sigma*100:.2f} cm")
    print(f"mean                 {readings.mean()*100:.2f} cm")
    print(f"sd (measured)        {readings.std(ddof=1)*100:.2f} cm")
    print(f"fraction within 1 sd {within_1sd:.4f}  (Gaussian predicts 0.6827)")
    print("Saved plot and summary to", output)

    if show:
        plt.show()
    else:
        plt.close("all")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=0.02,
                        help="sensor standard deviation in metres (default 0.02)")
    parser.add_argument("--trials", type=int, default=30_000)
    parser.add_argument("--no-show", action="store_true", help="save without opening a window")
    args = parser.parse_args()
    run(sigma=args.sigma, trials=args.trials, show=not args.no_show)
