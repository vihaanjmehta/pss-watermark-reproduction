# PSS Mini-Reproduction

I'm a high school student, and this is my attempt to actually *understand* the
Pattern Stability Score (PSS) from the paper **"Toward Resilient Watermark
Detection: Stability-Aware Statistical Features for Machine-Generated Text"**
(Mansouri, Marvania, and Safikhani).

Let me be upfront about scope, because it matters: **this is not a
reproduction of the paper.** The real paper runs on actual LLMs (LLaMA-2), a
real book corpus (PG-19), and real paraphrases produced by a language model. I
can't run any of that on my machine, so I built a much smaller, fully synthetic
version. It only implements the *core idea* of PSS — checking whether watermark
evidence stays stable when a text gets paraphrased — and isolates it well enough
that I could study it at home. None of the numbers here come from the paper, and
I'm not presenting them as if they did.

## Motivation

The word that grabbed me in the paper was **"stability."** A normal watermark
detector reads some text and counts how many tokens look "green" (the ones the
watermark secretly nudged the model toward). That works on one clean sample. But
real text gets paraphrased — people rewrite it, agents rewrite it — and
paraphrasing shuffles green tokens around. The paper's argument is that you
shouldn't trust a single snapshot. Instead, paraphrase the text several times
and ask: *does the evidence keep showing up, or was it a fluke?* Evidence that
survives every paraphrase is trustworthy; evidence that appears once and then
vanishes is probably noise.

I wanted to see that behavior happen with my own eyes, in the simplest setting I
could build. So my "documents" are just 240-bit sequences where position 1 means
"this token looks green." Mostly the green rate is 25%, but in a specific region
it jumps to a higher probability — that's the watermark hiding in the text. A
non-watermarked document is just those sequences with no polluted region at all.

## What I implemented

Since everything here is about the statistics, I tried to keep the code short
and commented enough that I (or a professor) can follow what each line does.

- **`synthetic_data.generate_sequence`** — makes a synthetic binary "document."
  Every position is an independent coin flip: `25%` green normally, a higher
  rate (like `70%`) inside the hidden signal region (positions 90–150).
- **`synthetic_data.make_paraphrase_depths`** — copies the sequence 4 times,
  each copy randomly flipping ~5% of the bits. Depth 0 is the original, depths
  1–3 are progressively more perturbed. This is a **toy stand-in for
  paraphrase**: it's definitely not a real paraphraser, it just gives me a way
  to simulate "what if the text got rephrased."
- **`pss.rolling_z_scores`** — slides a window across the sequence and, for each
  window, asks "how surprising is the green count I see, assuming the true green
  rate is 25%?" The answer is a z-score:
  `(observed count − expected count) ÷ √(window · 0.25 · 0.75)`. A big positive
  number means strong local watermark evidence.
- **`pss.pattern_stability_score`** — computes those z-scores for every
  paraphrase depth, then at each window position takes the **standard deviation
  of the z-scores across the 4 depths**. That standard deviation is my
  simplified PSS. Small std = the evidence is stable across paraphrases; large
  std = it wobbles.
- **`pss.detection_score`** — combines strength and stability into one number:
  `mean local z / (1 + PSS)`, taking the best window in the sequence. Dividing
  by `1 + PSS` punishes unstable evidence, which is the paper's whole point.
  Full honesty: this score is **my own made-up heuristic** so the study can be
  evaluated with ROC-AUC. It is not the classifier the paper uses.

## Experiments

I ran two experiments. Each condition uses 250 watermarked + 250
non-watermarked synthetic documents, and I report ROC-AUC (how well the
detection score tells the two apart; 1.0 = perfect, 0.5 = random).

1. **Window-size sensitivity** — signal probability fixed at `0.70`, and I
   varied the window: `{5, 10, 20, 30, 40, 50}`.
2. **Signal-strength sensitivity** — window fixed at `20`, and I varied how
   strong the hidden signal was: probabilities `{0.40, 0.50, 0.60, 0.70, 0.80,
   0.90}`.

The non-watermarked examples use a uniform `25%` rate everywhere, so the *only*
difference between the two classes is the hidden signal region.

## Results

These are the real outputs from `src/run_experiment.py` (fixed seed `42`,
250 per class per condition). I did not hand-wave or edit any of them.

**Experiment 1 — window size (signal probability = 0.70)**

| window size | ROC-AUC |
|------------:|--------:|
| 5           | 0.9314  |
| 10          | 0.9870  |
| 20          | 0.9980  |
| 30          | 0.9989  |
| 40          | 0.9994  |
| 50          | 0.9998  |

**Experiment 2 — signal strength (window size = 20)**

| signal probability | ROC-AUC |
|-------------------:|--------:|
| 0.40               | 0.7867  |
| 0.50               | 0.9222  |
| 0.60               | 0.9833  |
| 0.70               | 0.9992  |
| 0.80               | 0.9994  |
| 0.90               | 0.9997  |

Two things I noticed while looking at these:

1. **Bigger windows make detection much easier.** AUC climbs steadily from
   0.93 at window 5 and basically maxes out by window 20. That makes sense:
   a bigger window averages out the random noise in the green count, so the
   signal stands out more. At window 5 the detector is jumpy and can get thrown
   off by a single lucky window.
2. **There's a real detectability threshold for the signal.** At strength 0.40
   the signal is only ~1.5 z-units above the background and detection is weak
   (AUC 0.79). From 0.60 on, it's basically clean. So there is a floor: the
   watermark has to be strong enough to poke through the noise before
   stability-based detection can do its job.

I want to be careful about *how meaningful* these high AUCs are. Near-1.0
values are what you'd expect on my synthetic setup, and they'd be flattering if
I implied they say anything about real text. My simulator gives the detector
huge advantages a real system wouldn't have: it knows the exact null rate
(`gamma = 0.25`), the signal is always in the same place with the same length,
the sequence has no text-like patterns, and the "adversary" is random bit flips
rather than a paraphraser trying to actively erode the signal. So the *directions*
of these two trends matter, not the absolute numbers.

## Key Finding

In this synthetic setting, detection performance increased with window size,
while weaker watermark-like signals produced substantially lower AUC. That
suggests the amount of local context can strongly influence stability-based
detection — which I'd want to probe next under more realistic text and actual
paraphrasing, exactly the gap this toy setup can't fill.

## Limitations (the honest part)

* The "documents" are **synthetic binary sequences**, not machine-generated
  text. "Green bias" is just a higher coin-flip probability.
* The "paraphrases" are **random bit flips**. They don't model length changes,
  reordering, or meaning — a real language-model paraphrase does all three.
* I do **not** reproduce the paper's dataset, model, or pipeline.
* I make **no claim to reproduce the paper's headline results**. This is an
  independent implementation of one idea (PSS) with a detection score I
  invented for the study.

## Reproducibility

Everything uses a fixed random seed (`42` in `run_experiment.py`, `2026` in
`visualize_example.py`), so the same commands should give the same numbers on
any machine.

```bash
python3 -m pip install -r requirements.txt

python3 src/run_experiment.py      # runs both experiments, writes the CSVs and sensitivity figures
python3 src/visualize_example.py   # writes the local-signal and PSS example figures
```

## Reference

Sina Mansouri, Mohit Marvania, and Abolfazl Safikhani, *"Toward Resilient
Watermark Detection: Stability-Aware Statistical Features for Machine-Generated
Text."* OpenReview: https://openreview.net/forum?id=lIr8kHs8gI