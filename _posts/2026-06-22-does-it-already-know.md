---
layout: post
title: "When a model is wrong, does it already know better? Recovering suppressed answers from transfer-stable features"
date: 2026-06-22
---

# When a model is wrong, does it already know better?

*A frontier follow-up to [the reliable-feature account](/2026/06/22/reliable-feature-account/). That post decoded **why** a sentiment model fails out-of-distribution: spurious topic features (firing on the new register) override its genuine sentiment features. This post asks the deeper question that opens up — when the model's output is wrong, is the correct answer still **there**, inside the model, suppressed? The answer, measured: most of the time, yes. That's an eliciting-latent-knowledge result, and it reframes what an out-of-distribution error even is.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b + Gemma Scope SAE · 6 OOD domains, 897 errors · reproduced, with controls*

---

## The question

When a model gets an out-of-distribution answer wrong, there are two very different stories:

1. **Ignorance** — the model doesn't represent the right answer at all.
2. **Readout corruption** — the model *does* represent the right answer internally, but something overwrites it on the way to the output.

These have opposite implications for safety. If it's ignorance, you need a better model. If it's corruption, **the knowledge is already in there** and the job is to *read it out* — which is the eliciting-latent-knowledge (ELK) problem. So: which is it?

## The test

From the [previous post](/2026/06/22/reliable-feature-account/): the model's sentiment judgment runs on two kinds of features — **transfer-stable concept features** (genuine valence) and **fragile topic features** (subject-matter detectors that fire spuriously off-distribution and override the concept features). That gives a clean way to ask the question.

Treat the **transfer-stable subspace as a latent-knowledge probe.** Look only at the predictions the *full* model gets **wrong** out-of-distribution. On those errors, how often does a probe restricted to the model's **robust core** recover the *correct* answer?

- If recovery ≈ 50% (chance), the robust core didn't know either → ignorance.
- If recovery ≫ 50%, the correct answer was present in the model's stable features all along → corruption.

## The result

Pooled over **897 out-of-distribution errors** across six domains (amazon, yelp, imdb, tweets, poems, financial headlines):

| | recovery on the full model's OWN errors |
|---|---:|
| **transfer-stable core** | **61.5%** (95% CI 58.3–64.7%) |
| random-feature core (control) | 48.9% (≈ chance) |
| disjoint *half* of the stable core | 63.1% |

The robust core recovers the correct answer on **~62% of the cases the model got wrong** — significantly above the 50% chance floor, while random features recover nothing. And it isn't one lucky feature set: a *disjoint half* of the stable subspace recovers just as much, so the suppressed-correct answer is **distributed across the model's transfer-stable features.**

The effect is **strongest exactly where the output is weakest.** On financial headlines, where the full model sits at near-chance accuracy (0.50–0.59), the robust core still recovers **~69%** of its errors. The model looks lost at the output, but its concept features are mostly still right.

## What this says about what the model is thinking

> A large fraction of this model's out-of-distribution errors are **not ignorance — they are readout corruption.** The correct sentiment is represented in the transfer-stable features; the wrong answer comes out because spurious topic features (the new register's subject-matter detectors) drown that representation out. The model, in a real sense, *already knew*.

Combined with the [feature decoding](/2026/06/22/reliable-feature-account/#what-the-model-is-actually-thinking-reading-the-features), the full mechanistic story is now: *genuine valence features compute the right answer; topic detectors fire on the new domain's subject matter and override them; but the right answer remains linearly decodable from the stable subspace 62% of the time.* That's a complete, testable account of an OOD error as a **suppressed-but-present** correct computation.

## Honest scope and caveats

- The rigorous statement is "**the correct answer is linearly decodable from the transfer-stable subspace** on ~62% of the model's OOD errors," not "the model consciously knows." Probing-based latent-knowledge claims carry the standard caveat: decodability is not proof the model *uses* that information downstream.
- It's partial (62%, not 95%) and not uniform — one of twelve conditions (imdb, max-pool) fell to ~40%. Honest average, not a clean law.
- One model, one layer, sentiment, six domains. The claim is at that scope.

## Why this is the frontier direction

"Does the model already know the answer it failed to give?" is exactly the ELK question, and here it has a concrete, reproducible, controlled handle: **use the transfer-stable feature subspace as the latent-knowledge probe, and measure recovery on the model's own errors.** The next step is scale — instruct models and safety concepts, and generation rather than classification (does a hallucinating model's stable subspace still encode the truth it didn't say?). That's the program in the [proposal](https://github.com/OE-GOD/sae-feature-realness/blob/main/PROPOSAL_does_interp_know.md). Reproduction: [`latent_knowledge.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/latent_knowledge.py).
