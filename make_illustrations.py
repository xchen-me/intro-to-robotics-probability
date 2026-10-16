"""Conceptual setup drawings for the four demos - schematic, not photographic.

Author: Xu Chen

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
REPO_URL = "https://github.com/xchen-me/intro-to-robotics-probability"


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
    fig, ax = new_fig(13.0, 4.6)
    camera(ax, 0.4, 2.0, s=1.7, color=PURPLE)
    ax.plot([2.6, 2.6], [0.3, 4.2], color=GREY, lw=14, alpha=0.18, zorder=0)
    ax.text(2.6, 4.4, "a wall that never moves", ha="center", fontsize=17, color=GREY)
    px = Rectangle((2.35, 2.35), 0.46, 0.46, linewidth=2.8, edgecolor=GOLD, facecolor="none", zorder=5)
    ax.add_patch(px)
    ax.annotate("one pixel", xy=(2.6, 2.58), xytext=(5.6, 3.6), color=GOLD, fontsize=16,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.5))
    readings = [87, 85, 89, 86, 88, 90, 84]
    for i, v in enumerate(readings):
        ax.text(6.6 + 0.98 * i, 1.95 + 0.15 * np.sin(i), str(v), fontsize=15,
                color=TEAL if i % 2 else PURPLE, ha="center")
    ax.annotate("", xy=(6.3, 1.95), xytext=(4.6, 2.55),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.6))
    ax.text(9.9, 0.9, r"$X = \mu + \varepsilon$   (same patch of wall, 300 different numbers)",
            fontsize=17, color=GREY, ha="center")
    save(fig, "demo1_setup")


# ---------------------------------------------------------------- demo 2 ---
def demo2():
    fig, ax = new_fig(13.0, 4.6)
    ax.text(3.0, 4.05, "random noise", color=TEAL, fontsize=19, ha="center", weight="bold")
    for i, dx in enumerate((-1.1, 0.0, 1.1)):
        ax.plot(3.0 + dx, 2.65, marker="o", ms=13, color=TEAL, alpha=0.55 + 0.15 * (i % 2))
    ax.annotate("", xy=(3.0, 1.45), xytext=(3.0, 2.2),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.2))
    ax.text(3.0, 1.05, r"averages away: $\sigma/\sqrt{n}$", color=TEAL, fontsize=16, ha="center")

    ax.text(10.0, 4.05, "fixed-pattern offset", color=GOLD, fontsize=19, ha="center", weight="bold")
    for dx in (-1.1, 0.0, 1.1):
        ax.plot(10.0 + dx, 2.65, marker="o", ms=13, color=GOLD, alpha=0.85)
    ax.annotate("", xy=(10.0, 1.45), xytext=(10.0, 2.2),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.2))
    ax.text(10.0, 1.05, "survives every frame", color=GOLD, fontsize=16, ha="center")

    ax.plot([6.5, 6.5], [0.6, 4.3], color=GREY, lw=1.6, ls="--")
    ax.text(6.5, 4.45, "256 frames, averaged", color=GREY, fontsize=16, ha="center")
    save(fig, "demo2_setup")


# ---------------------------------------------------------------- demo 3 ---
def demo3():
    fig, ax = new_fig(13.0, 5.0)
    cx, cy, r = 6.5, 2.75, 2.0
    theta = np.linspace(0.35 * np.pi, 2.25 * np.pi, 120)
    ax.plot(cx + r * np.cos(theta), cy + 0.75 * r * np.sin(theta), color=GREY, lw=2.0,
            solid_capstyle="round")
    arrow_t = [0.55 * np.pi, 1.55 * np.pi]
    for t in arrow_t:
        x0, y0 = cx + r * np.cos(t), cy + 0.75 * r * np.sin(t)
        dx, dy = -np.sin(t) * r, np.cos(t) * 0.75 * r
        n = np.hypot(dx, dy)
        ax.annotate("", xy=(x0 + 0.01 * dx / n, y0 + 0.01 * dy / n), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=2.3))
    ax.text(cx - r - 0.75, cy, "predict\n(ellipse grows)", color=PURPLE, fontsize=17,
            ha="center", va="center")
    ax.text(cx + r + 0.75, cy, "correct\n(ellipse shrinks)", color=TEAL, fontsize=17,
            ha="center", va="center")

    ax.add_patch(Ellipse((2.6, 0.75), 0.7, 0.38, edgecolor=PURPLE, facecolor="none", lw=2.6))
    ax.plot(2.6, 0.75, marker="o", ms=7, color=PURPLE)
    ax.add_patch(Ellipse((10.4, 0.75), 1.9, 1.08, edgecolor=PURPLE, facecolor=PURPLE, alpha=0.12, lw=2.6))
    ax.plot(10.4, 0.75, marker="o", ms=7, color=PURPLE)
    ax.text(2.6, 0.1, "tracking: small ellipse", color=GREY, fontsize=14, ha="center")
    ax.text(10.4, 0.1, "covered: ellipse inflates", color=GREY, fontsize=14, ha="center")
    save(fig, "demo3_setup")


def demo3_scene():
    """The physical scene, simply: a camera watching one object move along a
    path, tracked frame by frame."""
    TANGERINE = "#E8622A"
    fig, ax = new_fig(11.5, 4.6)
    camera(ax, 0.3, 1.9, s=1.5, color=PURPLE)

    # A gentle arc for the object's path, left to right.
    t = np.linspace(0, 1, 200)
    path_x = 3.3 + 7.0 * t
    path_y = 2.6 + 1.1 * np.sin(2.3 * t)
    ax.plot(path_x, path_y, color=GREY, lw=1.6, ls=(0, (5, 4)), zorder=1)

    # Faint ghost positions behind the object, then the current position.
    for frac, alpha in ((0.25, 0.22), (0.5, 0.38), (0.72, 0.6)):
        i = int(frac * (len(t) - 1))
        ax.add_patch(Circle((path_x[i], path_y[i]), 0.42, facecolor=TANGERINE,
                            edgecolor="none", alpha=alpha, zorder=2))

    cx, cy = path_x[-1], path_y[-1]
    ax.add_patch(Circle((cx, cy), 0.46, facecolor=TANGERINE, edgecolor="#8a3f1a",
                        lw=1.8, zorder=4))
    ax.add_patch(Ellipse((cx, cy), 0.95, 0.7, edgecolor=PURPLE, facecolor="none",
                         lw=2.2, zorder=5))
    ax.plot(cx, cy, marker="o", ms=5, color=PURPLE, zorder=6)

    ax.text((3.3 + 10.3) / 2, 0.6, "tracking a moving object", color=GREY,
            fontsize=15, ha="center")
    save(fig, "demo3_scene")


# ---------------------------------------------------------------- demo 4 ---
def demo4():
    fig, ax = new_fig(13.0, 4.7)
    rng = np.random.default_rng(0)
    xs = np.linspace(-1.4, 1.4, 160)
    cols, rows = 5, 3
    for i in range(rows):
        for j in range(cols):
            cx = 1.4 + j * 2.4
            cy = 0.9 + i * 1.3
            sd = 0.22 + 0.18 * rng.random()
            ys = np.exp(-0.5 * (xs / sd) ** 2)
            ax.plot(cx + xs * 0.68, cy + ys * 0.52, color=PURPLE, lw=1.8, alpha=0.85)
            ax.plot([cx - 0.92, cx + 0.92], [cy, cy], color=GREY, lw=0.8, alpha=0.4)
    ax.text(6.8, 4.35, r"every pixel keeps its own $N(\mu_{ij},\,\sigma_{ij}^2)$",
            color=GREY, fontsize=18, ha="center")
    save(fig, "demo4_setup")


# ----------------------------------------------------------------- repo ---
def repo_qr():
    """QR code linking to the standalone student repo. Optional - only needs
    the 'qrcode' package, which nothing else here depends on."""
    try:
        import qrcode
    except ImportError:
        print("skipped assets/repo_qr.png (pip install qrcode[pil] to generate it)")
        return
    img = qrcode.make(REPO_URL, border=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_M)
    img = img.convert("RGB")
    pixels = img.load()
    purple = tuple(int(PURPLE[i:i + 2], 16) for i in (1, 3, 5))
    for y in range(img.height):
        for x in range(img.width):
            if pixels[x, y] == (0, 0, 0):
                pixels[x, y] = purple
    img.save(OUT / "repo_qr.png")
    print("assets/repo_qr.png")


if __name__ == "__main__":
    demo1()
    demo2()
    demo3()
    demo3_scene()
    demo4()
    repo_qr()
