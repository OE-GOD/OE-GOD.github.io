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

## A causal caveat: present, but not recoverable by subtraction

"Readout corruption" suggests a clean fix: if topic features merely *add* a wrong push, deleting them from the model's existing readout should restore the right answer. I tested that causally — and **it doesn't work.** Zeroing the least-transfer-stable features at the input to the *original* probe leaves financial accuracy flat (~0.50–0.58), no better than zeroing the same number of *random* features; remove too many and it degrades. The correct answer is recoverable only by **retraining** a readout on the stable subspace (the 62% result above), not by *subtracting* the unstable features from the original one.

So the precise statement is: the latent answer is **present and decodable, but entangled** — the model's readout mixes stable and unstable features in a way you can't cleanly undo by ablation. That's a concrete, small illustration of *why eliciting latent knowledge is hard*: you can't just delete the bad part; you need a new way to read. ([`causal_ablation.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/causal_ablation.py).)

## How to dig it out: the readout is the lever

If subtraction doesn't work, what does? I tried two routes.

**Erasure (failed, twice).** Identify the topic/domain subspace and project it out, then read sentiment. Both the naive version and a refined one (erase only the part of topic orthogonal to sentiment) **destroyed** the signal — accuracy fell to near chance (0.50–0.63). Topic and concept are entangled tightly enough that any direction you remove takes the answer with it. Erasure is the wrong tool.

**A shift-robust readout (worked).** The fix is not to *remove* features but to *read* the stable ones differently. Swapping the logistic probe for a **difference-of-class-means** readout on the transfer-stable features — a readout known to be more robust under distribution shift — digs out markedly more, exactly where the model is most broken:

| near-chance financial | model output | logistic on stable feats | **difference-of-means on stable feats** |
|---|---:|---:|---:|
| max-pool | 0.588 | 0.817 | **0.929** (CI 0.90–0.96) |
| mean-pool | 0.504 | 0.687 | **0.767** (CI 0.72–0.82) |

The improvement over the logistic readout is significant (paired bootstrap P = 1.00 / 0.996), the predictions are class-balanced (not a degenerate one-class trick), and it's identical across threshold choices. On a domain where the model's own answer is a coin flip, **a difference-of-means readout on its stable features recovers the right sentiment 93% of the time.** So the latent answer is not just present — with the right readout it is *substantially* recoverable. The constructive lesson for eliciting latent knowledge under shift: **select transfer-stable features, and read them with a shift-robust estimator (mass-mean), not a logistic probe — and don't try to erase the distractor.** ([`dig_out_knowledge.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/dig_out_knowledge.py), [`verify_massmean.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/verify_massmean.py).)

## Does it generalize to the model's own factual errors? No — and that's the boundary

The sentiment result is about a *probe's* errors. The real prize is the model's **own** errors during generation — hallucination. So I ran it: base Gemma-2-2b few-shot-**judges** ~750 factual True/False claims across five topics (cities, companies, animals, inventions, elements), with natural error rates (65–86% accuracy). When the *model itself* judges a claim wrong, does its internal state still encode the truth — recoverable by the same method (mass-mean on transfer-stable features, trained on held-out topics)?

**Mostly it doesn't — and I checked the artifacts the literature says fool people.** A first pass (layer-12 SAE features, mean-pooled) recovered truth on the model's errors *below* chance. But the truth-probing literature (Marks & Tegmark; Orgad et al.; Bürger et al.) names specific ways that can be an artifact, so I swept them on the raw residual stream across layers:

- **Token position mattered a lot** (Orgad et al.): at the **exact-answer token**, factual truth is decodable far better than mean-pooled — **AUROC 0.85 at layer 15** (vs ~0.70 pooled). So the model *does* linearly represent factual truth; my first pass was reading the wrong place. Correction logged.
- **But recovery on the model's errors still fails** where truth is best decoded (layer 15: error-set AUROC 0.52, ≈ chance). The apparently-high recovery at early layers occurs exactly where overall decoding is near-chance, and is untrustworthy: the model's errors are **91% accepted-false-claims** (an acceptance bias — it endorses false statements), leaving only ~12 true-claim errors to score. There is no credible signal that the model recovers the truths it gets wrong.
- **Metacognition also fails:** the best internal correctness-probe (AUROC 0.63) does **not** beat the model's own output confidence (0.64) at predicting its errors.

So, artifact-checked, the boundary holds — and it's sharper:

> **"The model knows more than it says" held for sentiment but fails for factual recall.** Sentiment has a *transferable concept* (valence) that survives the shift, so errors are spurious-feature interference and the answer is recoverable. For facts, the model *does* represent truth in general (0.85) — but on the specific claims it gets wrong it holds a **genuine false belief**, not suppressed knowledge: the "associated-hallucination" regime (Cheang et al., 2025), where a confident false belief built from real associations is mechanistically identical to correct recall, hence undetectable. And it can't internally flag those errors beyond its own confidence.

That matters for safety: interpretability here catches a *specific* failure — **spurious-feature interference with a robust concept** — but not confident false beliefs. It does *not* license "interpretability catches hallucinations." (Honest caveats: 2B model — linear truth structure is weak below ~13B, Marks & Tegmark; the acceptance bias leaves too few true-claim errors to settle recovery cleanly; base model + few-shot, ~150 claims/topic. Scripts: [`collect_factual.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/collect_factual.py), [`analyze_factual.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/analyze_factual.py), [`collect_factual_tokens.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/collect_factual_tokens.py), [`analyze_tokens.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/analyze_tokens.py).)

## Honest scope and caveats

- The rigorous statement is "**the correct answer is linearly decodable from the transfer-stable subspace** on ~62% of the model's OOD errors," not "the model consciously knows." Probing-based latent-knowledge claims carry the standard caveat: decodability is not proof the model *uses* that information downstream.
- It's partial (62%, not 95%) and not uniform — one of twelve conditions (imdb, max-pool) fell to ~40%. Honest average, not a clean law.
- One model, one layer, sentiment, six domains. The claim is at that scope.

## Why this is the frontier direction

"Does the model already know the answer it failed to give?" is exactly the ELK question, and here it has a concrete, reproducible, controlled handle: **use the transfer-stable feature subspace as the latent-knowledge probe, and measure recovery on the model's own errors.** The next step is scale — instruct models and safety concepts, and generation rather than classification (does a hallucinating model's stable subspace still encode the truth it didn't say?). That's the program in the [proposal](https://github.com/OE-GOD/sae-feature-realness/blob/main/PROPOSAL_does_interp_know.md). Reproduction: [`latent_knowledge.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/latent_knowledge.py).
