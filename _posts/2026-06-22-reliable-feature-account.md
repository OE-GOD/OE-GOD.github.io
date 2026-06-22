---
layout: post
title: "What makes an out-of-distribution answer untrustworthy — and when interpretability can tell"
date: 2026-06-22
---

# What makes an out-of-distribution answer untrustworthy — and when interpretability can tell

*This is the synthesis the [agreement-signal post](/2026/06/21/interpretability-beats-confidence/) was reaching for. After publishing that, I ran four large automated fleets — to invent better signals, to attack my own result, to diagnose my own mistakes, and to prove the theory — and along the way collected a genuinely harder distribution shift. What survived is not "disagreement beats confidence." It's a sharper, partly-proven, honestly-bounded account of **when** interpretability can tell a model is wrong, and **why**. It corrects two overclaims I'd have published otherwise.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b + Gemma Scope SAE · sentiment, 6 OOD domains · every number below reproduced by me, not taken from an agent*

---

## The one-line account

> Trust an out-of-distribution prediction **to the degree that transfer-stable features carry it.** A prediction built on features that don't survive the shift is the untrustworthy one — and you can measure that without any labels.

Everything below is that sentence, made precise, with the cases where it holds, the case where something else wins, and the mechanism underneath.

## The setup

A sentiment probe over 16,384 Gemma Scope SAE features, accurate in-distribution, tested on six out-of-distribution domains: three review shifts (amazon, yelp, imdb) and three genuinely different registers I collected to stress it — **tweets** (informal), **poems** (literary), and **financial headlines** (formal). Financial is the key case: the probe lands at **near-chance there (accuracy 0.50–0.59)**. We want a label-free *trust signal* that separates the probe's correct OOD predictions from its wrong ones.

## Three things that are now nailed down

**1. Plain confidence cannot be repaired, and it collapses exactly when you need it.**
There's a clean theorem here: a detector's error-*ranking* (its AUROC) is invariant under any global monotone rescaling of the score — so temperature/Platt calibration is a no-op for this purpose. Confidence's ranking power is fixed. And empirically it *collapses under shift*: across the six domains, the correlation between the detector's accuracy and confidence's error-ranking AUROC is **+0.87** — confidence only knows it's wrong when the model is already mostly right. On near-chance financial, confidence is useless (AUROC 0.51–0.60).

**2. The working signal is "is this call backed by transfer-stable features?" — and the mechanism is a better oracle, not magic.**
Restrict the probe to its ~500 transfer-stable ("reliable") features. That restricted probe is a **strictly better out-of-distribution classifier**: mean accuracy **0.815 vs 0.723** for the full probe (random-feature restriction: 0.62). On financial it's 0.69–0.82 vs 0.50–0.59. So the trust signal isn't two coins failing independently — it's *comparing the deployed model to a more shift-robust version of itself.* When the full model's call rests on shift-fragile features and the reliable view objects, the call is probably wrong. Directly verified: on financial, **wrong** predictions have reliable features pushing **against** them (−0.4 to −1.1), while **correct** ones have reliable features backing them (+0.8 to +1.4).

**3. The best operationalization is a single-probe evidence attribution.**
You don't even need the second probe. Score a prediction by how much of *its own margin* comes from reliable features: `sign(margin) × (reliable features' contribution to the margin)`. This **reliable-aligned evidence** signal beats confidence decisively (domain-clustered bootstrap AUROC gap **+0.234, P = 1.00**) and edges the refit-based "surprisal" signal on the hard shift (**+0.053, P = 0.98**), with its biggest win exactly where it matters — **financial AUROC 0.92–0.97.** Disagreement, surprisal, and evidence-attribution all correlate ~0.6: three implementations of the same principle, the simplest one best.

## The result that matters most for safety: graceful degradation

| detector state | reliable-evidence AUROC | confidence AUROC | gap |
|---|---:|---:|---:|
| **failing** (accuracy < 0.65, incl. near-chance financial) | 0.83 | 0.55 | **+0.28** |
| okay (accuracy ≥ 0.75) | 0.85 | 0.72 | +0.13 |

The interpretability signal's advantage over confidence **doubles as the model gets worse.** On near-chance financial, you can't fix the prediction — but you can still reliably *gate* it (lifting selective accuracy 0.50→0.83+ at matched coverage). A monitor whose edge grows in the regime you actually care about is the property you want from a safety tool.

## The correction: this is not a universal law (and I almost published it as one)

My instinct was "two-view disagreement is all you need; density/novelty never helps." A fleet built to *prove* that instead **refuted it**, and I reproduced the counterexample:

- Under an **extrapolation regime** — where a model's accuracy degrades smoothly with distance from its training support — plain **marginal density is a valid error signal**, AUROC **0.81**, with no class contrast at all (correct points sit at radius 1.96, wrong at 2.73).
- It fails on *my* shifts (AUROC 0.48–0.60, often anti-predictive) only because those shifts are **boundary-dominated**: the whole domain moves off-manifold, so "unusual" encodes *register distance*, not *wrongness*.

So the honest statement is **regime-conditional**: density tells you about error exactly when novelty is positively coupled to error (extrapolation, open-set), and the two-view/reliable-feature account dominates when the shift is boundary-dominated and a transfer-stable subspace stays accurate. There's even a label-free diagnostic to tell which regime you're in: regress novelty on the model's margin over the unlabeled test batch — trust density only if novel points are systematically lower-margin.

## Two more honest limits

- **Ranking, not operating points.** Everything above is about *ordering* correct above wrong (AUROC). Turning that into a label-free keep/abstain *threshold* with a coverage guarantee does **not** survive shift — split-conformal coverage drifts to 0.42–0.58 against a nominal 0.50. You can rank trust without labels; you cannot get distribution-free coverage without them.
- **Scope.** One model, one layer, sentiment, six domains, small n (the two poolings per domain are correlated, so the honest unit is ~6 domains, not 12 conditions). The claims that survive are stated at that scope.

## Why I think this is the real shape of the thing

The arc went: *SAE features mostly aren't real → validate detectors → can't make them accurate OOD → make them abstain → confidence does the abstaining, not interpretability → no, a reliable-feature signal beats confidence → and here's the principle, the mechanism, the best form, the safety property, and the exact boundary where it stops being true.* Each step was forced by attacking the previous one. The two corrections in this post (density can work; the mechanism is "better oracle," not "independent failure") came from fleets I built specifically to break my own claims — and they did.

What's genuinely established, narrowly and with proof where possible: **for boundary-dominated shifts, the trustworthiness of a model's answer is carried by whether transfer-stable features support it; this beats confidence robustly and most where the model is weakest; and it is one regime of a larger picture, not a law.** Reproductions: [`verify_reliable_evidence.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/verify_reliable_evidence.py), [`mechanism_check.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/mechanism_check.py), [`frontier_accuracy_vs_trust.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/frontier_accuracy_vs_trust.py), [`hard_harness.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/hard_harness.py).

The next real step is scale — instruct models, safety concepts, and shifts that are *extrapolation* rather than register, where the theory predicts the picture changes. That needs more than a laptop, and it's exactly what the [research proposal](https://github.com/OE-GOD/sae-feature-realness/blob/main/PROPOSAL_does_interp_know.md) is now scoped to do.
