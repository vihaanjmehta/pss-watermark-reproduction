# PSS Mini-Reproduction

An independent, small-scale re-implementation and sensitivity study of the
**Pattern Stability Score (PSS)** idea, inspired by the paper *"Toward
Resilient Watermark Detection: Stability-Aware Statistical Features for
Machine-Generated Text"*. This repository **does not** reproduce that paper's
results, dataset, or pipeline. It implements the core *concept* — measuring
how local watermark evidence behaves across perturbed versions of a document —
on controlled synthetic data, and then studies how window size and signal
strength affect detection.

## Motivation

A watermark detector works by finding statistical evidence (e.g. an excess of
"green" tokens) that a language model deliberately left in its output. A naive
detector reads test sets that were all generated at one point in time. The
paper's insight is that *real* test sets vary: users paraphrase, agents rewrite,
and the same prompt may be re-rolled. A statistic that stays **stable** across
such variations is more trustworthy than one that spikes in a single draw.

This project isolates that single idea: the local z-score of a sequence is
computed in a rolling window, and its **standard deviation across perturbation
depths** is used as a stability-aware feature. Everything here is a toy binary
abstraction, so the behavior of the statistic itself can be studied under a
known ground truth.

## What I Implemented

* **Rolling local z-scores** (`pss.rolling_z_scores`) — for each sliding
  window, the positive-count is standardized against the expected count and
  binomial variance under the null rate `gamma = 0.25`.
* **Perturbation-depth sequences** (`synthetic_data.make_paraphrase_depths`) —
  `depth 0` is the original sequence; each later depth applies one more binary
  perturbation pass, a toy stand-in for progressively heavier paraphrasing.
* **Window-wise Pattern Stability Score** (`pss.pattern_stability_score`) —
  per window position, the standard deviation of the local z-scores across
  perturbation depths (stability) together with their mean (strength).
* **Detection score + ROC-AUC** (`pss.detection_score`) — a simple,
  interpretable statistic `mean_z / (1 + pss)` evaluated with
  `sklearn.metrics.roc_auc_score`. This is an original scalar heuristic for
  this mini-study, **not** the classifier used in the paper.

## Experiments

Both experiments generate 250 watermarked and 250 non-watermarked synthetic
documents per condition and evaluate the detection score with ROC-AUC.

1. **Window-size sensitivity** — signal probability fixed at `0.70`; window
   sizes `{5, 10, 20, 30, 40, 50}`.
2. **Signal-strength sensitivity** — window size fixed at `20`; signal
   probabilities `{0.40, 0.50, 0.60, 0.70, 0.80, 0.90}`.

Non-watermarked documents use `signal_prob = background_prob = 0.25`
(uniform), so the only difference between the classes is the elevated signal
region.

## Results

Results below are the actual outputs written by `src/run_experiment.py`
(fixed seed `42`, 250 positive / 250 negative examples per condition).
They are **not** estimates or edits.

### Experiment 1 — window-size sensitivity (signal probability = 0.70)

| window size | ROC-AUC |
|------------:|--------:|
| 5           | 0.9314  |
| 10          | 0.9870  |
| 20          | 0.9980  |
| 30          | 0.9989  |
| 40          | 0.9994  |
| 50          | 0.9998  |

### Experiment 2 — signal-strength sensitivity (window size = 20)

| signal probability | ROC-AUC |
|-------------------:|--------:|
| 0.40               | 0.7867  |
| 0.50               | 0.9222  |
| 0.60               | 0.9833  |
| 0.70               | 0.9992  |
| 0.80               | 0.9994  |
| 0.90               | 0.9997  |

CSVs: `results/window_size_results.csv`, `results/signal_strength_results.csv`.
Figures: `figures/window_size_sensitivity.png`, `figures/signal_strength_sensitivity.png`,
plus `figures/local_signal_example.png` and `figures/pss_example.png` from
`src/visualize_example.py`.

## Limitations

* The "documents" are **synthetic binary sequences**, not machine-generated
  text; "green" bias is modeled as an elevated Bernoulli probability.
* The "perturbations" are **random bit flips**, not language-model paraphrases.
  They do not model length changes, reordering, or semantics.
* This does not reproduce the paper's complete dataset, model, or pipeline.
* This **does not claim to reproduce the paper's headline results**; it is an
  independent implementation of the PSS *concept* used for a small sensitivity
  study with a self-invented, simple detection score.

## Reproducibility

All randomness uses a fixed seed (`SEED = 42` in `run_experiment.py`, `2026`
in `visualize_example.py`).

```bash
python3 -m pip install -r requirements.txt

python3 src/run_experiment.py   # runs both experiments, writes CSVs + sensitivity figures
python3 src/visualize_example.py  # writes the local-signal and PSS example figures
```

## Reference

Zhiyuan Wang, et al., *"Toward Resilient Watermark Detection: Stability-Aware
Statistical Features for Machine-Generated Text."*
OpenReview: https://openreview.net/forum?id=lIr8kHs8gI