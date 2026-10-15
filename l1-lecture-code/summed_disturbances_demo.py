"""Lecture 1: the CLT on a non-uniform, bimodal starting distribution.

Author: Xu Chen

Setup: a 50/50 mixture of Uniform[-1, -.3] and Uniform[.1, 1.4]; 40,000
trials, 24 independent disturbances summed per trial.
Assumptions: IID terms with finite variance. Centering and scaling use the
sample mean and sample standard deviation of one disturbance, not a
closed-form formula, since a mixture's moments are easiest to read off the
data itself.
Takeaway: a single disturbance is bimodal - nothing like a bell curve - yet
the standardized sum of 24 of them is already close to N(0,1). The CLT does
not care that the parent distribution looks nothing like a Gaussian.

Run from any directory: python path/to/summed_disturbances_demo.py [--no-show]
Requires NumPy and Matplotlib. Saves a plot and a JSON summary to
modernized_lectures/code-results/l1.
"""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
N_TERMS = 24


def one_disturbance(rng, size):
    """50/50 mixture of two uniform pieces - deliberately bimodal."""
    left = rng.uniform(-1.0, -0.3, size=size)
    right = rng.uniform(0.1, 1.4, size=size)
    pick_left = rng.random(size) < 0.5
    return np.where(pick_left, left, right)


def simulate(trials=40_000, seed=2026):
    rng = np.random.default_rng(seed)
    single = one_disturbance(rng, trials)
    m, s = single.mean(), single.std(ddof=0)
    terms = one_disturbance(rng, (trials, N_TERMS))
    total = terms.sum(axis=1)
    standardized = (total - N_TERMS * m) / (s * np.sqrt(N_TERMS))
    return single, standardized, m, s


def run(trials=40_000, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = ROOT / "code-results/l1"
    output.mkdir(parents=True, exist_ok=True)

    single, standardized, m, s = simulate(trials)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    a1.hist(single, bins=60, density=True, color="#4B2E83", alpha=0.6)
    a1.set_title("one disturbance (bimodal)")
    a1.set_xlabel("value")
    a1.set_ylabel("density")

    grid = np.linspace(-4, 4, 400)
    phi = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)
    a2.hist(standardized, bins=80, density=True, range=(-4, 4),
            color="#4B2E83", alpha=0.6, label="standardized sum, n=24")
    a2.plot(grid, phi, color="#9B761F", lw=2, label=r"$N(0,1)$")
    a2.set_title("standardized sum of 24 disturbances")
    a2.set_xlabel(r"$z$")
    a2.legend(fontsize=9)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "summed_disturbances.png", dpi=180)
    fig.savefig(output / "summed_disturbances.pdf")

    summary = dict(trials=trials, n_terms=N_TERMS,
                   single_mean=float(m), single_std=float(s),
                   standardized_mean=float(standardized.mean()),
                   standardized_std=float(standardized.std(ddof=0)))
    (output / "summed_disturbances_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"single disturbance    mean {m:.4f}  sd {s:.4f}")
    print(f"standardized sum (n={N_TERMS}) mean {standardized.mean():.4f}  "
          f"sd {standardized.std(ddof=0):.4f}")
    print("Saved plot and summary to", output)

    if show:
        plt.show()
    else:
        plt.close("all")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40_000)
    parser.add_argument("--no-show", action="store_true", help="save without opening a window")
    args = parser.parse_args()
    run(trials=args.trials, show=not args.no_show)
