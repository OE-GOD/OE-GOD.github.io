---
layout: post
title: "When interpretability beats confidence: distrust the answer your trustworthy self rejects"
date: 2026-06-21
---

# When interpretability beats confidence: distrust the answer your trustworthy self rejects

*A follow-up to [the abstention capstone](/2026/06/21/trustworthy-by-abstention/), which ended on an honest catch: a detector could tell when it was wrong out-of-distribution, but the working signal was plain confidence — the interpretability-native signal I tried failed at chance. This post is the sharpened version, and it flips that catch. A better interpretability signal — **does the detector agree with the version of itself that uses only its trustworthy features?** — beats plain confidence at knowing-when-wrong, across three held-out domains and two poolings, and a control proves the interpretability is doing the work.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b + Gemma Scope SAE · sentiment, real classifier*

---

## The idea in one line

Build the detector twice: once on **all** its features, once on only its **trustworthy** (transfer-stable) features. Run both on each input. **When the two disagree, don't trust the answer.**

The intuition: if the full detector predicts "positive" but the trustworthy-features-only version predicts "negative," the full one is leaning on unreliable features that don't transfer — so it's probably wrong out-of-distribution.

## Why this, and why now

The [capstone](/2026/06/21/trustworthy-by-abstention/) showed you can make a detector trustworthy OOD by abstaining on low-confidence inputs — but the signal that worked was the classifier's own confidence, and the interpretability-native signal I tried (abstain when the input activates training-rare features) failed at chance (AUROC 0.51). That left the field's central hope unanswered: *does interpretability add anything to knowing-when-wrong, beyond plain confidence?*

The naive interpretability signal failed because it was naive. This is the sharpened one.

## The result

At matched coverage, paired bootstrap, leak-free (trustworthy features chosen from SST-2 + rotten_tomatoes only; the OOD test domains never used to build the signal):

| pooling | domain | plain confidence | **reliable-agreement** | random-feature agreement (control) |
|---|---|---:|---:|---:|
| max | amazon | 0.750 | **0.836** | 0.746 |
| max | yelp | 0.799 | **0.858** | 0.778 |
| max | imdb | 0.787 | **0.827** | 0.803 |
| mean | amazon | 0.773 | **0.857** | 0.787 |
| mean | yelp | 0.748 | **0.883** | 0.756 |
| mean | imdb | 0.771 | **0.807** | 0.754 |

(Numbers are selective accuracy — accuracy on the inputs the signal chooses to keep — at the agreement signal's natural coverage, with plain confidence held to the same coverage.)

**Reliable-agreement beats plain confidence on all six conditions**, and the paired bootstrap puts P(agreement > confidence) at 0.99–1.00. Abstaining on the disagreement set raises OOD accuracy from ~0.70–0.76 to ~0.81–0.88.

## The control that matters: interpretability is load-bearing

A skeptic's first objection: *"this is just disagreement between two models — generic ensembling, nothing to do with interpretability."* So I ran the control — the second model on **random** 500 features instead of the trustworthy ones (5 seeds):

- **Random-feature agreement ≈ plain confidence** (0.75–0.80) — generic ensembling adds little.
- **Reliable-feature agreement beats it by +0.02 to +0.13** on every condition.

So it isn't "ask two models." It's *"ask the version of yourself that only uses trustworthy features"* — and **choosing the trustworthy features (the interpretability step) is what makes the signal work.** That's the answer to the open question: yes, interpretability adds value to knowing-when-wrong, over both plain confidence and generic ensembling.

## Honest scope and framing

- It is, mechanically, a form of **sub-model disagreement** — the contribution is that the sub-model is defined by an interpretability criterion (transfer-stable "reliable" features), and the control shows that criterion is what carries the gain.
- Sentiment, one model, three OOD domains, two poolings. Verified hard (matched coverage, paired significance, leak-free, load-bearing control) — but modest in scope. The natural next step is scale: instruct models and safety concepts, per the [research proposal](https://github.com/OE-GOD/sae-feature-realness/blob/main/PROPOSAL_does_interp_know.md).

## Where this leaves the arc

The capstone said: *trustworthy ≠ accurate; you get OOD trust by knowing when to abstain — but plain confidence, not interpretability, was doing it.* This post updates that:

> **Interpretability does beat plain confidence at knowing-when-wrong — once you use it to identify trustworthy features and distrust predictions your trustworthy self rejects.** The naive interpretability signal failed; the sharpened one wins, and a control confirms the interpretability is load-bearing.

That turns the whole arc's ending from "selective prediction works, interpretability didn't help" into "interpretability *does* help — here's the signal, and here's the control proving it." It's the first verified case in this program where an interpretability-native method beats the simple baseline it had to dethrone.
