---
layout: post
title: "Placing a result: where 'interpretability beats confidence' sits in the literature — and the paper that's its biggest threat"
date: 2026-06-21
---

# Placing a result: where "interpretability beats confidence" sits in the literature — and the paper that's its biggest threat

*A follow-up to [the agreement-signal post](/2026/06/21/interpretability-beats-confidence/). That post made a positive claim: an interpretability-native signal beats plain confidence at knowing-when-wrong out-of-distribution. Before building anything on top of it, I did two things a result this size deserves — I reproduced it from scratch as a saved script, and I read the field hard enough to find where it actually sits and what most threatens it. This post is that reckoning. It is mostly related work and honesty, not new wins.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b + Gemma Scope SAE · sentiment, real classifier*

---

## First: it reproduces

The [previous post's](/2026/06/21/interpretability-beats-confidence/) result — distrust a prediction when the full detector disagrees with the version of itself that uses only transfer-stable features — was, embarrassingly, never saved as a script. It lived in throwaway inline code. A thorough review of my own work flagged exactly that: the headline result *was not in the repository*, so nobody (including me) could re-run it.

So I rebuilt it from scratch in [`verify_agreement_signal.py`](https://github.com/OE-GOD/sae-feature-realness). It reproduces to the third decimal:

| pooling | domain | plain confidence | reliable-agreement | random control |
|---|---|---:|---:|---:|
| max | amazon | 0.750 | **0.836** | 0.746 |
| max | yelp | 0.799 | **0.858** | 0.778 |
| max | imdb | 0.787 | **0.827** | 0.803 |
| mean | amazon | 0.773 | **0.857** | 0.787 |
| mean | yelp | 0.748 | **0.883** | 0.756 |
| mean | imdb | 0.771 | **0.807** | 0.754 |

Reliable-agreement beats confidence on all six conditions; the random-feature control ties confidence, not reliable. The result stands — and now it is reproducible, which it wasn't before. That a review of my own work caught a phantom-until-saved result is the point: the check you most need is the one that tries to find your result and fails.

## Where it actually sits

The honest version of "is this novel?" is "novel *compared to what?*" — so I read the two literatures it touches.

**Disagreement-for-reliability.** [Agreement-on-the-Line (Baek, Jiang, Raghunathan, Kolter, NeurIPS 2022)](https://arxiv.org/abs/2206.13089) and the line of work around it show that *disagreement between two models* tracks generalization error under distribution shift — remarkably well. This is strong external support that an agreement-based signal *should* work. But their two models are **independently trained**, and they predict **aggregate** OOD error, not per-instance trust.

**Interpretability-native abstention.** [Ferrando, Obeso, Rajamanoharan, Nanda — "Do I Know This Entity?" (ICLR 2025)](https://arxiv.org/abs/2411.14257) use SAE feature directions on Gemma Scope to detect when a model doesn't recognize an entity, and causally drive refusal. Internal interpretable features *do* carry an actionable "should I trust this?" signal.

These two lines stay separate. The cell between them — **disagreement between a model and a mechanistically-defined restriction of *itself*, per instance, with a control proving the interpretability (not just added variance) is doing the work** — is the one the agreement signal occupies. The second "model" isn't an independent network; it's the same detector restricted to its transfer-stable features. That's the honest novelty: a *variant* of a known idea, plus a control the disagreement literature usually skips.

## The paper that's the biggest threat

The most important thing the literature read turned up isn't supporting — it's threatening, and that's why it matters most.

[Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda — "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing" (ICML 2025)](https://arxiv.org/abs/2502.16681) tested SAEs as probes across four regimes including **covariate shift**, and found they **do not reliably beat simple baselines**. They specifically demand a baseline/control comparison — the exact control I ran.

I'm not going to pretend this doesn't apply. It does. My claim has to be narrow and orthogonal to survive it:

> SAE structure may be load-bearing for **selective risk** (knowing when to abstain) even in a regime where it isn't load-bearing for **accuracy**.

Those are different targets. Kantamneni et al. is about accuracy; the agreement signal is about abstention. The random-feature control is the test that decides it. And I'll pre-commit now: **if a scaled version of this signal does not beat the random control, that is a confirmation of Kantamneni et al. in a new metric — and I'll report it as such, not bury it.** A result that can only come out one way isn't a result.

## Two corrections to how I'd been measuring and pitching

- **Metric.** The right way to score selective prediction is [AUGRC (Traub, Bungert, Lüth, …, Jaeger, NeurIPS 2024)](https://arxiv.org/abs/2407.01032), not the older AURC, which can misrank methods. The scaled study should report AUGRC + full risk-coverage curves, so the work lands cleanly as selective prediction.
- **Baseline.** Plain confidence ([Geifman & El-Yaniv, NeurIPS 2017](https://arxiv.org/abs/1705.08500)) is the champion, and it won at AUROC ~0.70 here. Beating it means beating that *exact number*, not a strawman — their own principle is that you only beat confidence with a *different signal source*. Between-hypothesis disagreement is that source.

## The sharpened method, stated honestly

The scaled form — call it **Reliable-Feature Disagreement (RFD)** — keeps the signal and removes its rough edges:

1. **No second training run.** One detector, refit on its transfer-stable feature subset. Disagreement between the two is the signal.
2. **Parameter-free core:** abstain iff the full and reliable predictions differ.
3. **A stability-ranked coverage knob** so the threshold isn't a free parameter — abstain first on inputs whose label hinges on the least-transferable features.
4. **The random-feature control is the whole scientific point**, not a footnote. If RFD doesn't beat it, interpretability isn't the active ingredient.

I want to be exact about status: the laptop result above is *reproduced*. RFD-at-scale — instruct models, safety concepts, multiple OOD domains, AUGRC — is a **proposal**, not a result. This post earns the proposal by placing it against the real field, including the work most likely to sink it.

## What this post is, and isn't

It isn't a new experimental win. It's the unglamorous step between a laptop result and a real research program: reproduce it so it exists, find where it sits, name the paper that threatens it, fix the metric, and pre-commit to reporting the null. The full deepened design lives in [the repo's proposal](https://github.com/OE-GOD/sae-feature-realness/blob/main/PROPOSAL_does_interp_know.md).

The arc so far: *features mostly aren't real → validate detectors instead → you can't make them accurate OOD → but you can make them abstain → and interpretability beats plain confidence at it.* This post adds the part that makes the last claim trustworthy: **here's exactly how novel it is, and here's the paper I most have to beat.**
