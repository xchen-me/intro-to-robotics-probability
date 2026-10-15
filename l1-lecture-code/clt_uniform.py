"""Lecture 1: sum independent Uniform[0, 1] variables, then standardize.

Author: Xu Chen

Setup: 100,000 independent trials; each row contains 12 independent draws.
Assumptions: IID uniform variables, mean 1/2, variance 1/12, no shared bias.
Takeaway: S_n broadens as n grows; Z_n=(S_n-n/2)/sqrt(n/12) approaches N(0,1).
The n=2 sum is exactly triangular, not exactly Gaussian. Finite histograms vary.

Run from any directory: python path/to/clt_uniform.py [--no-show]
Requires NumPy and Matplotlib. Saves plots, histogram CSVs and summary JSON in
modernized_lectures/code-results/l1. No SciPy or notebook is required.
"""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SIZES = (1, 2, 3, 12)


def simulate(trials=100_000, seed=2026):
    """Return paired raw/standardized sums and checks using analytic moments."""
    rng = np.random.default_rng(seed)
    draws = rng.uniform(0.0, 1.0, size=(trials, max(SAMPLE_SIZES)))
    # cumsum preserves the pairing: column n-1 contains X_1 + ... + X_n.
    cumulative = np.cumsum(draws, axis=1)
    samples, summary = {}, []
    for n in SAMPLE_SIZES:
        total = cumulative[:, n - 1]
        standardized = (total - n / 2) / np.sqrt(n / 12)
        samples[n] = (total, standardized)
        summary.append(dict(n=n, mean=float(total.mean()),
                            variance=float(total.var(ddof=0)),
                            theoretical_mean=n / 2, theoretical_variance=n / 12,
                            standardized_mean=float(standardized.mean()),
                            standardized_variance=float(standardized.var(ddof=0))))
    return samples, summary


def run(show=True):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = ROOT / 'code-results/l1'
    output.mkdir(parents=True, exist_ok=True)
    samples, summary = simulate()
    grid = np.linspace(-4, 4, 401)
    phi = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
    # Each panel uses density normalization: sum(height * bin_width) = 1.
    # Histogram height is NOT the probability of a single exact real value.
    for label, index in [('raw', 0), ('standardized', 1)]:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
        for ax, n in zip(axes, SAMPLE_SIZES):
            values = samples[n][index]
            edges = np.linspace(0, n, 101) if index == 0 else np.linspace(
                -np.sqrt(3 * n), np.sqrt(3 * n), 101)
            heights, edges = np.histogram(values, bins=edges, density=True)
            assert np.isclose(np.sum(heights * np.diff(edges)), 1.0)
            centers = (edges[:-1] + edges[1:]) / 2
            np.savetxt(output / f'{label}_{n}.csv',
                       np.column_stack((centers, heights)), delimiter=',',
                       header='x,density', comments='')
            ax.stairs(heights, edges, fill=True, alpha=.55, color='#4B2E83',
                      label='Simulated density')
            if index == 1:
                ax.plot(grid, phi, color='#9B761F', label=r'$\mathcal{N}(0,1)$')
                ax.set_xlim(-4, 4)
            elif n == 2:
                exact_x = np.linspace(0, 2, 201)
                ax.plot(exact_x, np.maximum(0, 1 - abs(exact_x - 1)),
                        color='#9B761F', label='Exact triangular density')
            ax.set_title(rf'$n={n}$')
            ax.set_xlabel(r'$s$' if index == 0 else r'$z$')
            ax.set_ylabel(r'$\widehat{p}_{S_n}(s)$' if index == 0
                          else r'$\widehat{p}_{Z_n}(z)$')
            ax.spines[['top', 'right']].set_visible(False)
        axes[-1].legend(fontsize=8, loc='upper right')
        fig.savefig(output / f'clt_{label}.pdf')
        fig.savefig(output / f'clt_{label}.png', dpi=180)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    for row in summary:
        print(f"n={row['n']:2d}: mean={row['mean']:.5f} "
              f"(theory {row['theoretical_mean']:.5f}), "
              f"variance={row['variance']:.5f} "
              f"(theory {row['theoretical_variance']:.5f})")
    print('Saved plots and numerical data to', output)
    if show:
        plt.show()
    else:
        plt.close('all')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true', help='Save without opening windows')
    run(show=not parser.parse_args().no_show)
