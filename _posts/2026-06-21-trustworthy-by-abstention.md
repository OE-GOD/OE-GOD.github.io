---
layout: post
title: "Trustworthy not by being right, but by knowing when it's wrong"
date: 2026-06-21
---

# Trustworthy not by being right, but by knowing when it's wrong

*Capstone of a four-post arc on whether you can trust what interpretability tells you about a model. The thread: SAE features mostly aren't real, so you validate detectors instead; but those detectors don't generalize across distributions, and every lever I tried to fix that — de-spuriousing, a generalization predictor, sign-stability selection, multi-domain training — gave small, inconsistent, or no gains. This post is the resolution. I stopped trying to make the detector accurate out-of-distribution and asked a different question: can it know when it's wrong and abstain? That works — out-of-distribution accuracy rises from ~0.73 to ~0.83 by abstaining on the least-confident half, consistently across three held-out domains, verified against a random-abstention baseline.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b + Gemma Scope SAE · sentiment, real classifier*

---

## The wall

The first three posts kept hitting one wall: a sentiment detector built from SAE features works in-distribution (~0.87 F1) but drops to 0.66–0.74 on a new domain (product reviews, restaurant reviews, long-form reviews). I tried, in order:

- **de-spuriousing** (drop features that encode the domain) — real, but [subsumed by recipe choice](/2026/06/20/despurious-cross-distribution/);
- a **generalization predictor** (predict OOD-usefulness from in-distribution properties) — r≈0.49, and it *lost* to naive feature selection;
- **sign-stability selection** (keep features whose sentiment-sign is consistent across two in-dist domains) — +0.05, but significant on only 2 of 3 held-out domains;
- **multi-domain training** — worse on all three.

The honest pattern: **making an SAE-feature detector more *accurate* out-of-distribution is hard, and the simple baseline is stubbornly strong.** Eleven methods, no reliable win.

## The reframe

So I stopped trying to make it accurate, and asked the question that actually matters for safety:

> A trustworthy detector isn't one that's always right. It's one that **knows when it's wrong** — and abstains.

A detector that says "I don't know" out-of-distribution is safe even when it can't be accurate. That's *selective prediction* ([Geifman & El-Yaniv](https://arxiv.org/abs/1705.08500)), pointed at the OOD-trust problem. The test: is there a confidence signal that separates the detector's correct OOD predictions from its wrong ones — so you can answer on the confident ones and abstain on the rest?

I tested five signals across three held-out domains (amazon, yelp, imdb) and two poolings:

| signal | mean AUROC (correct vs wrong) | selective-accuracy gain | works? |
|---|---:|---:|---|
| margin / probability | 0.68 | **+0.107** | yes, 6/6 configs |
| ensemble disagreement | 0.60 | +0.073 | unstable |
| Mahalanobis-to-train | 0.53 | +0.011 | no |
| feature-novelty (interp-native) | 0.51 | +0.003 | no |

## The result

**It works.** Abstaining on the least-confident 50% raises out-of-distribution accuracy from **~0.73 to ~0.83**, and it holds across all three held-out domains and both poolings (positive in 6/6 configurations). Verified: random 50% abstention keeps accuracy at ~0.75 (as it must), while the confidence-ranked version reaches 0.83–0.87 — well outside the random band.

So the detector *can* know when it's about to be wrong out-of-distribution, and you can convert that into a more trustworthy system by letting it abstain.

## The honest catch

The signal that works is **plain classifier confidence** (the margin / softmax probability) — **not** the interpretability-native one. I expected the SAE-feature signal — "abstain when the input activates features that were rare in training" — to win, because it's the interpretable story. It failed at chance (AUROC 0.51). Distance-to-training-distribution (Mahalanobis) was equally useless (0.53). Only the classifier's own confidence knew when it was wrong.

So the accurate claim is: **selective prediction makes SAE-feature detectors trustworthy out-of-distribution — via standard confidence, not a special interpretability signal.** A known technique, applied to this problem, verified; the interpretability angle I hoped for did not pan out.

## What the whole arc adds up to

> **You cannot (yet) make an SAE-feature detector accurate out-of-distribution — but you can make it trustworthy out-of-distribution, by having it abstain when its own confidence is low.** Accuracy OOD is hard and the obvious levers don't help; calibrated abstention OOD works (0.73 → 0.83, three domains, verified).

That reframes the whole question. Across the arc, "trustworthy interpretability signal" kept failing when defined as *accuracy*. Defined as *knowing when not to trust it*, it's achievable. And it points at the right design for anything built on these signals — including interpretability-driven control or self-improvement: a signal that **abstains out-of-distribution** is the guard against the confident-but-wrong failure (the reward-hacking I saw when [a self-improvement loop gamed its own signal](/2026/06/20/detect-and-steer-frontier/)).

## Honest scope

- Sentiment, one model, three OOD domains, a real classifier. The effect is consistent but modest (+0.10 selective accuracy at 50% coverage).
- The working mechanism is generic confidence (selective prediction is a known method); the interpretability-native abstention signal did not work here.
- "Accuracy OOD is hard" is established across ~11 methods on this setup; it is not a claim that no method anywhere can do it — only that the obvious ones don't, and that abstention is the achievable form of trust.
