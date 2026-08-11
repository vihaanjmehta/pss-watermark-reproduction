"""Run both sensitivity experiments and write results + figures.

Usage::

    python3 src/run_experiment.py

The script:

* Experiment 1 (window size): window sizes [5, 10, 20, 30, 40, 50],
  250 watermarked + 250 non-watermarked synthetic examples each, signal
  probability fixed at 0.70.
* Experiment 2 (signal strength): signal probabilities
  [0.40, 0.50, 0.60, 0.70, 0.80, 0.90], window size fixed at 20,
  250 watermarked + 250 non-watermarked examples each.

For every condition the per-document ``detection_score`` is computed and
evaluated with ROC-AUC (scikit-learn).  Outputs are written to
``results/*.csv`` and ``figures/*.png``.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pss import detection_score
from synthetic_data import (
    DEFAULT_BACKGROUND_PROB,
    DEFAULT_SIGNAL_PROB,
    generate_sequence,
    make_paraphrase_depths,
)

# Fixed seed so the whole run is reproducible on any machine.
SEED = 42
NUM_EXAMPLES = 250  # 250 positive + 250 negative per condition

WINDOW_SIZES = [5, 10, 20, 30, 40, 50]
SIGNAL_PROBS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
FIXED_WINDOW = 20

# Paths are resolved relative to the repository root so the script works
# regardless of which directory it is invoked from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")


def generate_scores(
    rng, signal_prob, window_size, n_examples=NUM_EXAMPLES
):
    """Return detection scores for `n_examples` synthetic documents.

    Parameters
    ----------
    rng : numpy.random.Generator
        Single shared generator (keeps the whole run deterministic).
    signal_prob : float
        Positive probability inside the signal region.  Setting it equal to
        ``background_prob`` yields non-watermarked examples.
    window_size : int
        Rolling window size used for the detection score.
    n_examples : int
        Number of independent documents to draw.

    Returns
    -------
    numpy.ndarray
        ``n_examples`` document-level detection scores.
    """
    scores = np.empty(n_examples)
    for i in range(n_examples):
        sequence = generate_sequence(
            signal_prob=signal_prob,
            background_prob=DEFAULT_BACKGROUND_PROB,
            rng=rng,
        )
        depths = make_paraphrase_depths(sequence, rng=rng)
        scores[i] = detection_score(depths, window_size)
    return scores


def compute_auc(positive_scores, negative_scores):
    """ROC-AUC for the binary classification positive-vs-negative."""
    labels = np.concatenate(
        [np.ones(len(positive_scores)), np.zeros(len(negative_scores))]
    )
    scores = np.concatenate([positive_scores, negative_scores])
    return float(roc_auc_score(labels, scores))


def run_window_size_experiment(rng):
    """Experiment 1: how window size affects detection at signal prob 0.70."""
    rows = []
    print("\n=== Experiment 1: window size (signal probability = 0.70) ===")
    for window_size in WINDOW_SIZES:
        positive = generate_scores(rng, DEFAULT_SIGNAL_PROB, window_size)
        negative = generate_scores(rng, DEFAULT_BACKGROUND_PROB, window_size)
        auc = compute_auc(positive, negative)
        rows.append({"window_size": window_size, "auc": auc})
        print(f"  window_size={window_size:>3d}  ROC-AUC={auc:.4f}")
    return pd.DataFrame(rows)


def run_signal_strength_experiment(rng):
    """Experiment 2: how signal strength affects detection at window 20."""
    rows = []
    print("\n=== Experiment 2: signal strength (window size = 20) ===")
    for signal_prob in SIGNAL_PROBS:
        positive = generate_scores(rng, signal_prob, FIXED_WINDOW)
        negative = generate_scores(rng, DEFAULT_BACKGROUND_PROB, FIXED_WINDOW)
        auc = compute_auc(positive, negative)
        rows.append({"signal_probability": signal_prob, "auc": auc})
        print(f"  signal_probability={signal_prob:.2f}  ROC-AUC={auc:.4f}")
    return pd.DataFrame(rows)


def save_line_plot(x, y, xlabel, ylabel, title, filename):
    """Save a simple sensitivity line plot using default matplotlib colors."""
    fig, ax = plt.subplots(figsize=(7, 5))
    # No explicit colors are passed: matplotlib's default color cycle is used.
    ax.plot(x, y, marker="o", linestyle="-", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # One deterministic generator for every draw in both experiments.
    rng = np.random.default_rng(SEED)

    df_window = run_window_size_experiment(rng)
    df_signal = run_signal_strength_experiment(rng)

    csv_window = os.path.join(RESULTS_DIR, "window_size_results.csv")
    csv_signal = os.path.join(RESULTS_DIR, "signal_strength_results.csv")
    df_window.to_csv(csv_window, index=False)
    df_signal.to_csv(csv_signal, index=False)
    print(f"\nCSV written: {csv_window}")
    print(f"CSV written: {csv_signal}")

    save_line_plot(
        df_window["window_size"].to_numpy(),
        df_window["auc"].to_numpy(),
        xlabel="window size",
        ylabel="ROC-AUC",
        title="Window size sensitivity (signal probability = 0.70)",
        filename="window_size_sensitivity.png",
    )
    save_line_plot(
        df_signal["signal_probability"].to_numpy(),
        df_signal["auc"].to_numpy(),
        xlabel="signal probability (strength)",
        ylabel="ROC-AUC",
        title="Signal strength sensitivity (window size = 20)",
        filename="signal_strength_sensitivity.png",
    )
    print(f"Figure written: {os.path.join(FIGURES_DIR, 'window_size_sensitivity.png')}")
    print(f"Figure written: {os.path.join(FIGURES_DIR, 'signal_strength_sensitivity.png')}")


if __name__ == "__main__":
    main()