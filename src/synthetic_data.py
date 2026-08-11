"""Synthetic binary sequences used as simplified "watermark evidence".

This module provides two tools:

* ``generate_sequence`` -- a binary sequence with a localized region of
  elevated positive ("green") probability.
* ``perturb_sequence`` / ``make_paraphrase_depths`` -- toy perturbations
  that stand in for paraphrasing.

HONESTY / SCOPE
---------------
These sequences are NOT samples of machine-generated text, and the
perturbations are NOT language-model paraphrases.  They are deliberately
simple binary abstractions used only to study the *local statistics* that
the Pattern Stability Score (PSS) concept operates on.  This is a controlled
synthetic experiment so that known ground truth exists; it cannot say
anything about real watermark detectors.  See the README for the full list
of limitations.
"""

import numpy as np

# Default parameters used throughout the mini-study.
DEFAULT_LENGTH = 240
DEFAULT_SIGNAL_START = 90
DEFAULT_SIGNAL_END = 150
DEFAULT_BACKGROUND_PROB = 0.25
DEFAULT_SIGNAL_PROB = 0.70
DEFAULT_FLIP_PROB = 0.05


def generate_sequence(
    length=DEFAULT_LENGTH,
    signal_start=DEFAULT_SIGNAL_START,
    signal_end=DEFAULT_SIGNAL_END,
    background_prob=DEFAULT_BACKGROUND_PROB,
    signal_prob=DEFAULT_SIGNAL_PROB,
    rng=None,
):
    """Generate a binary sequence with a localized elevated-probability region.

    Each position is an independent Bernoulli draw:

    * ``P(positive = 1) = background_prob`` outside ``[signal_start, signal_end)``
    * ``P(positive = 1) = signal_prob``    inside  ``[signal_start, signal_end)``

    The elevated region is a toy stand-in for the tokens that a generative
    watermark colors as "green" (biased toward a watermark-derived random
    seed).  A "watermarked" document has ``signal_prob > background_prob``;
    a "non-watermarked" document has ``signal_prob == background_prob``.

    Parameters
    ----------
    length : int
        Number of positions in the sequence.
    signal_start, signal_end : int
        Half-open interval ``[start, end)`` containing the signal region.
    background_prob : float
        Base probability of a positive observation (the null rate).
    signal_prob : float
        Elevated positive probability inside the signal region.
    rng : numpy.random.Generator, optional
        A seeded random generator.  Pass one for reproducible results.

    Returns
    -------
    numpy.ndarray
        Integer array of 0/1 values with shape ``(length,)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build the Bernoulli parameter for every position, then draw once.
    probabilities = np.full(length, background_prob)
    probabilities[signal_start:signal_end] = signal_prob
    return (rng.random(length) < probabilities).astype(np.int64)


def perturb_sequence(sequence, flip_prob=DEFAULT_FLIP_PROB, rng=None):
    """Return a noisy copy of a binary sequence.

    Each position is independently flipped (0 <-> 1) with probability
    ``flip_prob``.  This is a *toy abstraction* of progressive paraphrase
    degradation: a paraphrased version should retain most of its statistical
    character while introducing some noise.

    IMPORTANT: this is NOT an actual LLM paraphraser.  Real paraphrases can
    change token order, length, and semantics, none of which are modeled
    here.  The abstraction is used only so that the effect of "perturbation
    depth" on local statistics can be studied under full control.

    Parameters
    ----------
    sequence : numpy.ndarray
        Binary sequence to perturb.
    flip_prob : float
        Probability that a single position is flipped.
    rng : numpy.random.Generator, optional

    Returns
    -------
    numpy.ndarray
        Binary array of the same length as `sequence`.
    """
    if rng is None:
        rng = np.random.default_rng()

    flip_mask = rng.random(len(sequence)) < flip_prob
    return np.where(flip_mask, 1 - sequence, sequence).astype(np.int64)


def make_paraphrase_depths(
    base_sequence,
    num_depths=4,
    flip_prob=DEFAULT_FLIP_PROB,
    rng=None,
):
    """Create stacked perturbation-depth versions of one sequence.

    Depth 0 is the original sequence.  Each subsequent depth applies one
    more perturbation pass on top of the previous depth, so higher depths
    are progressively more altered -- a toy analogue of more aggressive
    paraphrasing.  PSS will later summarize statistics *across* these depths.

    Parameters
    ----------
    base_sequence : numpy.ndarray
        Binary sequence to copy and perturb.
    num_depths : int
        Number of depth versions (>= 1).
    flip_prob : float
        Flip probability used for each perturbation pass.
    rng : numpy.random.Generator, optional

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(num_depths, len(base_sequence))``.
    """
    if rng is None:
        rng = np.random.default_rng()

    depths = [np.array(base_sequence, dtype=np.int64)]
    current = depths[0].copy()
    for _ in range(num_depths - 1):
        current = perturb_sequence(current, flip_prob=flip_prob, rng=rng)
        depths.append(current)
    return np.stack(depths)