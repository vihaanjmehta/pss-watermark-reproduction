"""Core statistics for the PSS mini-reproduction.

This module implements three pieces:

1. ``rolling_z_scores`` -- a rolling-window local z-score that measures how
   much a window's positive ("green") count deviates from the rate expected
   under the null distribution.
2. ``pattern_stability_score`` -- the window-wise standard deviation of the
   local z-scores *across perturbation depths*.  This is a simplified port of
   the "Pattern Stability Score" concept from the reference paper.
3. ``detection_score`` -- a small, interpretable scalar statistic used only
   for the sensitivity study in this repository.

SCOPE NOTE
----------
PSS is originally defined in the paper over real LLM outputs and real
paraphrase perturbations.  Everything here operates on the synthetic binary
sequences from ``synthetic_data.py``; it is an independent re-implementation
of the *idea* of local stability, not a reproduction of the paper's
pipeline or headline numbers.
"""

import numpy as np

# Baseline positive rate assumed for non-watermarked text ("green" rate).
# Must agree with synthetic_data.DEFAULT_BACKGROUND_PROB.
DEFAULT_GAMMA = 0.25


def rolling_z_scores(binary_sequence, window_size, gamma=DEFAULT_GAMMA):
    """Compute a rolling-window local z-score for a binary sequence.

    For each valid window (one window for every starting position ``i``
    such that ``i + window_size <= len(sequence)``):

    * count the positive observations ``c`` in the window;
    * compute the count expected under the null hypothesis ``gamma``;
    * standardize using the binomial variance

      .. math::
          z = \\frac{c - w \\cdot \\gamma}{\\sqrt{w \\cdot \\gamma \\cdot (1 - \\gamma)}}

    ``gamma`` is the reference "green/positive" rate assumed for
    non-watermarked text.  Under the null hypothesis (no watermark) each
    window's count is roughly Binomial ``(w, gamma)``, so ``z`` has mean
    ~0 and variance ~1.  A real watermark raises the true positive rate
    above ``gamma`` inside the signal region, pushing ``z`` strongly
    positive exactly there.

    Parameters
    ----------
    binary_sequence : numpy.ndarray
        Binary (0/1) observation sequence.
    window_size : int
        Number of consecutive observations per window.
    gamma : float
        Null positive rate (see ``synthetic_data.DEFAULT_BACKGROUND_PROB``).

    Returns
    -------
    numpy.ndarray
        One z-score per valid rolling window, shape ``(n - window_size + 1,)``.
        ``result[i]`` is the z-score of window ``[i, i + window_size)``.
    """
    sequence = np.asarray(binary_sequence, dtype=np.float64)
    n = len(sequence)

    if window_size < 1 or window_size > n:
        raise ValueError(f"window_size must be in [1, {n}], got {window_size}")

    # Cumulative sums let every window count be computed in O(1) instead of
    # O(n * window_size).
    cumulative = np.concatenate(([0.0], np.cumsum(sequence)))
    counts = cumulative[window_size:] - cumulative[:-window_size]

    expected = window_size * gamma
    variance = window_size * gamma * (1.0 - gamma)  # binomial variance under H0

    z_scores = (counts - expected) / np.sqrt(variance)
    return z_scores


def pattern_stability_score(sequences, window_size, gamma=DEFAULT_GAMMA):
    """Compute the window-wise Pattern Stability Score given depth sequences.

    Given several sequences that represent different perturbation depths of
    the *same* underlying document (see ``make_paraphrase_depths``):

    1. compute the rolling local z-score for every depth;
    2. stack these z-score curves into a ``(num_depths, num_windows)`` array;
    3. for each window position compute:

       * ``mean_z``   = mean of the local z-scores across depths
       * ``pss``      = standard deviation of the local z-scores across
         depths (the stability score)

    Intuition: in the signal region every paraphrase depth shows strong,
    *consistently* positive evidence, so ``std`` across depths is small and
    ``mean_z`` is large.  Outside the signal region the evidence is weak and
    behaves like correlated noise, so the score pattern differs.

    Parameters
    ----------
    sequences : array-like of shape (num_depths, length)
        Binary depth sequences sharing a common length.
    window_size : int
        Rolling window size passed to ``rolling_z_scores``.
    gamma : float
        Null positive rate.

    Returns
    -------
    pss : numpy.ndarray
        Standard deviation of the local z-scores across depths, per window.
    mean_z : numpy.ndarray
        Mean local z-score across depths, per window.
    """
    depth_z = np.stack(
        [rolling_z_scores(seq, window_size, gamma=gamma) for seq in sequences]
    )

    # np.std defaults to ddof=0 (population std).  With only 4 depths the
    # sample/population distinction matters little, but ddof=1 is the more
    # conventional sample standard deviation used for a stability estimate.
    pss = depth_z.std(axis=0, ddof=1)
    mean_z = depth_z.mean(axis=0)
    return pss, mean_z


def detection_score(sequences, window_size, gamma=DEFAULT_GAMMA):
    """Combine signal strength and stability into a document-level score.

    Per-window, we combine the depth-averaged local z-score (strength) with
    the PSS (instability) into:

    .. math::
        \\text{local evidence}_i = \\frac{\\overline{z}_i}{1 + \\text{pss}_i}

    The ``+1`` keeps the denominator strictly positive and dampens the boost
    from unstable (high-variance) windows: strong-but-*stable* evidence is
    rewarded.  The document score is the maximum local evidence over all
    windows -- the "strongest aligned evidence" peak statistic, which suits
    a localized signal region.

    HONESTY NOTE
    ------------
    This is a deliberately simple statistic invented for this mini-study so
    that ROC-AUC evaluation is transparent and easy to explain.  It is NOT
    claimed to be the classifier used in the reference paper.

    Parameters
    ----------
    sequences : array-like of shape (num_depths, length)
        Perturbation-depth versions of one document.
    window_size : int
        Rolling window size.
    gamma : float
        Null positive rate.

    Returns
    -------
    float
        Document-level detection score.
    """
    pss, mean_z = pattern_stability_score(sequences, window_size, gamma=gamma)
    local_evidence = mean_z / (1.0 + pss)
    return float(np.max(local_evidence))