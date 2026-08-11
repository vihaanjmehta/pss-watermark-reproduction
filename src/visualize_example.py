"""Produce example figures that visualize the local signal and PSS.

Usage::

    python3 src/visualize_example.py

Generates, for one fixed seed:

* ``figures/local_signal_example.png`` -- rolling local z-scores vs window
  position for a watermarked sequence (with its perturbation depths) against
  a non-watermarked companion sequence.
* ``figures/pss_example.png``     -- the window-wise Pattern Stability Score
  (std of local z-scores across perturbation depths) for the same pair.

The signal region ``[90, 150)`` is shaded so the different local behavior is
easy to see.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pss import pattern_stability_score, rolling_z_scores
from synthetic_data import (
    DEFAULT_BACKGROUND_PROB,
    DEFAULT_SIGNAL_END,
    DEFAULT_SIGNAL_PROB,
    DEFAULT_SIGNAL_START,
    generate_sequence,
    make_paraphrase_depths,
)

SEED = 2026
WINDOW_SIZE = 20

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "figures")


def depth_z_curves(sequences, window_size):
    """Rolling local z-scores for every depth, stacked as (depth, window)."""
    return np.stack([rolling_z_scores(seq, window_size) for seq in sequences])


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # One pair of documents: watermarked (elevated signal region) and
    # non-watermarked (uniform background rate everywhere).
    watermarked = generate_sequence(signal_prob=DEFAULT_SIGNAL_PROB, rng=rng)
    plain = generate_sequence(signal_prob=DEFAULT_BACKGROUND_PROB, rng=rng)

    wm_depths = make_paraphrase_depths(watermarked, rng=rng)
    plain_depths = make_paraphrase_depths(plain, rng=rng)

    wm_z = depth_z_curves(wm_depths, WINDOW_SIZE)
    plain_z = depth_z_curves(plain_depths, WINDOW_SIZE)

    wm_pss, wm_mean_z = pattern_stability_score(wm_depths, WINDOW_SIZE)
    plain_pss, plain_mean_z = pattern_stability_score(plain_depths, WINDOW_SIZE)

    # Window positions are the starting index of each window.
    positions = np.arange(wm_z.shape[1])

    # Colors are taken from matplotlib's default cycle (not hand-picked).
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ---- Figure 1: local z-score signal -----------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    # Individual perturbation-depth curves (faint, show the spread).
    ax.plot(positions, wm_z.T, color=colors[0], alpha=0.2, linewidth=1)
    ax.plot(positions, plain_z.T, color=colors[1], alpha=0.2, linewidth=1)
    # Depth-averaged local z-scores (the "strength of evidence" signal).
    ax.plot(
        positions,
        wm_mean_z,
        color=colors[0],
        linewidth=2.5,
        label="watermarked: mean local z across depths",
    )
    ax.plot(
        positions,
        plain_mean_z,
        color=colors[1],
        linewidth=2.5,
        label="non-watermarked: mean local z across depths",
    )
    ax.axhline(0, color=colors[3], linewidth=1, linestyle=":")
    ax.axvspan(DEFAULT_SIGNAL_START, DEFAULT_SIGNAL_END - WINDOW_SIZE, color=colors[0], alpha=0.12)
    ax.set_xlabel("window starting position")
    ax.set_ylabel("local z-score")
    ax.set_title("Rolling local z-score vs. window position (window = 20)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "local_signal_example.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 2: Pattern Stability Score --------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        positions,
        wm_pss,
        color=colors[0],
        linewidth=2,
        label="watermarked: PSS (std of z across depths)",
    )
    ax.plot(
        positions,
        plain_pss,
        color=colors[1],
        linewidth=2,
        label="non-watermarked: PSS (std of z across depths)",
    )
    ax.axvspan(DEFAULT_SIGNAL_START, DEFAULT_SIGNAL_END - WINDOW_SIZE, color=colors[0], alpha=0.12)
    ax.set_xlabel("window starting position")
    ax.set_ylabel("PSS (std of local z across perturbation depths)")
    ax.set_title("Pattern Stability Score vs. window position (window = 20)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "pss_example.png"), dpi=150)
    plt.close(fig)

    print("Wrote figures/local_signal_example.png")
    print("Wrote figures/pss_example.png")


if __name__ == "__main__":
    main()