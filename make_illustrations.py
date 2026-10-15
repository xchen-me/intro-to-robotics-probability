"""Conceptual setup drawings for the four demos - schematic, not photographic.

Same purple/gold/teal/grey palette as the lecture figures in ../../support and
as the demo scripts themselves, so the deck reads as one family of material.
These are drawn with matplotlib (already a dependency of the demos) rather
than the course's TikZ/pdflatex pipeline, to keep this package self-contained.

Run: python make_illustrations.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

PURPLE, GOLD, TEAL, GREY = "#4B2E83", "#B7A57A", "#2E7D9A", "#5F5F6E"
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def camera(ax, x, y, s=1.0, color=PURPLE):
    """Schematic camera body + lens, pointing right, with a three-leg tripod."""
    ax.add_patch(FancyBboxPatch((x, y), 1.0 * s, 0.7 * s,
                                 boxstyle="round,pad=0.015,rounding_size=0.08",
                                 linewidth=2, edgecolor=color, facecolor="white", zorder=3))
    ax.add_patch(Circle((x + 0.3 * s, y + 0.35 * s), 0.22 * s, linewidth=2,
                         edgecolor=color, facecolor="white", zorder=4))
    ax.add_patch(Circle((x + 0.3 * s, y + 0.35 * s), 0.09 * s, facecolor=color, zorder=4))
    foot = y
    for dx in (-0.35, 0.0, 0.5):
        ax.plot([x + 0.5 * s, x + 0.5 * s + dx * s], [foot, foot - 0.55 * s],
                 color=color, lw=1.4, zorder=2)
    return x + 1.0 * s, y + 0.35 * s  # lens-ish exit point for arrows


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08,
                transparent=True)
    plt.close(fig)
    print(f"assets/{name}.png")


# ---------------------------------------------------------------- demo 1 ---
def demo1():
    fig, ax = new_fig(10.5, 3.6)
    camera(ax, 0.3, 1.6, s=1.3, color=PURPLE)
    ax.plot([2.0, 2.0], [0.3, 3.3], color=GREY, lw=10, alpha=0.18, zorder=0)
    ax.text(2.0, 3.45, "a wall that never moves", ha="center", fontsize=11, color=GREY)
    px = Rectangle((1.82, 1.82), 0.36, 0.36, linewidth=2.2, edgecolor=GOLD, facecolor="none", zorder=5)
    ax.add_patch(px)
    ax.annotate("one pixel", xy=(2.0, 2.0), xytext=(4.3, 2.85), color=GOLD, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.2))
    readings = [87, 85, 89, 86, 88, 90, 84]
    for i, v in enumerate(readings):
        ax.text(5.2 + 0.78 * i, 1.55 + 0.12 * np.sin(i), str(v), fontsize=11,
                color=TEAL if i % 2 else PURPLE, ha="center")
    ax.annotate("", xy=(5.0, 1.55), xytext=(3.7, 2.0),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.3))
    ax.text(7.9, 0.75, r"$X = \mu + \varepsilon$   (same patch of wall, 300 different numbers)",
            fontsize=12, color=GREY, ha="center")
    save(fig, "demo1_setup")


# ---------------------------------------------------------------- demo 2 ---
def demo2():
    fig, ax = new_fig(10.5, 3.6)
    ax.text(2.4, 3.2, "random noise", color=TEAL, fontsize=13, ha="center", weight="bold")
    for i, dx in enumerate((-0.9, 0.0, 0.9)):
        ax.plot(2.4 + dx, 2.1, marker="o", ms=9, color=TEAL, alpha=0.55 + 0.15 * (i % 2))
    ax.annotate("", xy=(2.4, 1.15), xytext=(2.4, 1.75),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6))
    ax.text(2.4, 0.85, r"averages away: $\sigma/\sqrt{n}$", color=TEAL, fontsize=11, ha="center")

    ax.text(8.1, 3.2, "fixed-pattern offset", color=GOLD, fontsize=13, ha="center", weight="bold")
    for dx in (-0.9, 0.0, 0.9):
        ax.plot(8.1 + dx, 2.1, marker="o", ms=9, color=GOLD, alpha=0.85)
    ax.annotate("", xy=(8.1, 1.15), xytext=(8.1, 1.75),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.6))
    ax.text(8.1, 0.85, "survives every frame", color=GOLD, fontsize=11, ha="center")

    ax.plot([5.25, 5.25], [0.5, 3.4], color=GREY, lw=1.2, ls="--")
    ax.text(5.25, 3.5, "256 frames, averaged", color=GREY, fontsize=11, ha="center")
    save(fig, "demo2_setup")


# ---------------------------------------------------------------- demo 3 ---
def demo3():
    fig, ax = new_fig(10.5, 3.9)
    cx, cy, r = 5.25, 2.0, 1.55
    theta = np.linspace(0.35 * np.pi, 2.25 * np.pi, 120)
    ax.plot(cx + r * np.cos(theta), cy + 0.75 * r * np.sin(theta), color=GREY, lw=1.6,
            solid_capstyle="round")
    arrow_t = [0.55 * np.pi, 1.55 * np.pi]
    for t in arrow_t:
        x0, y0 = cx + r * np.cos(t), cy + 0.75 * r * np.sin(t)
        dx, dy = -np.sin(t) * r, np.cos(t) * 0.75 * r
        n = np.hypot(dx, dy)
        ax.annotate("", xy=(x0 + 0.01 * dx / n, y0 + 0.01 * dy / n), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.8))
    ax.text(cx - r - 0.55, cy, "predict\n(ellipse grows)", color=PURPLE, fontsize=12,
            ha="center", va="center")
    ax.text(cx + r + 0.55, cy, "correct\n(ellipse shrinks)", color=TEAL, fontsize=12,
            ha="center", va="center")

    ax.add_patch(Ellipse((2.1, 0.55), 0.55, 0.3, edgecolor=PURPLE, facecolor="none", lw=2))
    ax.plot(2.1, 0.55, marker="o", ms=5, color=PURPLE)
    ax.add_patch(Ellipse((8.4, 0.55), 1.5, 0.85, edgecolor=PURPLE, facecolor=PURPLE, alpha=0.12, lw=2))
    ax.plot(8.4, 0.55, marker="o", ms=5, color=PURPLE)
    ax.text(2.1, 0.08, "tracking: small ellipse", color=GREY, fontsize=10, ha="center")
    ax.text(8.4, 0.08, "covered: ellipse inflates", color=GREY, fontsize=10, ha="center")
    save(fig, "demo3_setup")


# ---------------------------------------------------------------- demo 4 ---
def demo4():
    fig, ax = new_fig(10.5, 3.7)
    rng = np.random.default_rng(0)
    xs = np.linspace(-1.4, 1.4, 160)
    cols, rows = 5, 3
    for i in range(rows):
        for j in range(cols):
            cx = 1.1 + j * 1.95
            cy = 0.75 + i * 1.05
            sd = 0.22 + 0.18 * rng.random()
            ys = np.exp(-0.5 * (xs / sd) ** 2)
            ax.plot(cx + xs * 0.55, cy + ys * 0.42, color=PURPLE, lw=1.3, alpha=0.85)
            ax.plot([cx - 0.75, cx + 0.75], [cy, cy], color=GREY, lw=0.6, alpha=0.4)
    ax.text(5.5, 3.45, r"every pixel keeps its own $N(\mu_{ij},\,\sigma_{ij}^2)$",
            color=GREY, fontsize=12, ha="center")
    save(fig, "demo4_setup")


if __name__ == "__main__":
    demo1()
    demo2()
    demo3()
    demo4()
