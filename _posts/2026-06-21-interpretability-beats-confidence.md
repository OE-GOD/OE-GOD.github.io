---
layout: post
title: "When interpretability beats confidence: distrust the answer your trustworthy self rejects"
date: 2026-06-21
---

# When interpretability beats confidence: distrust the answer your trustworthy self rejects

*A follow-up to [the abstention capstone](/2026/06/21/trustworthy-by-abstention/), which ended on an honest catch: a detector could tell when it was wrong out-of-distribution, but the working signal was plain confidence — the interpretability-native signal I tried failed at chance. This post is the sharpened version, and it flips that catch. A better interpretability signal — **does the detector agree with the version of itself that uses only its trustworthy features?** — beats plain confidence at knowing-when-wrong, across three held-out domains and two poolings, and a control proves the interpretability is doing the work.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b + Gemma Scope SAE · sentiment, real classifier*

*Updated June 2026 — I ran a 59-agent adversarial search to try to beat this signal. A graded version (reliable-code surprisal) wins on ranking, no non-disagreement mechanism beats it, and the fixed-coverage accuracy gain is a coverage artifact. See [the update](#update-june-2026-i-tried-to-beat-this-signal-with-59-agents) at the end.*

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

---

## Update (June 2026): I tried to beat this signal with 59 agents

After publishing, I ran two large automated searches — 59 agents total — to *attack* the agreement signal: invent and test better trust signals. Twenty-six candidates, every one run on the same real data through one fixed harness, with an independent leakage audit on every winner. Three things came out — one positive, one cautionary, and a third that matters more than both.

**1. A graded version wins on ranking.** The published signal is a single binary switch ("abstain when the full and reliable detectors disagree"). Two *graded* refinements rank correct-vs-wrong predictions strictly better across *all* coverage levels:

- **Reliable-code surprisal** (the new best) — trust a prediction by *how strongly the reliable-feature sub-model believes the full model's chosen label*: concretely, the reliable probe's log-probability of the full probe's predicted class. **AUROC 0.73 → 0.83** (higher in all six conditions), AUGRC better in all six, paired bootstrap **P ≈ 0.99–1.00**.
- A **bootstrap ensemble** of reliable sub-models (AUROC ~0.81) does nearly as well.

**2. The cautionary caveat — and it's the honest core.** I coverage-matched both against the binary champion. At the binary signal's own operating point, selective accuracy is an **exact tie — 0/6 improvement, Δ = 0.000.** The graded signals don't make the *kept* answers more accurate there; they give a better-*ranked* **dial** — pick any coverage and they separate right from wrong better at it. So any flashy "selective accuracy 0.86–0.94" is a **coverage artifact** (a continuous signal gets scored at ~50% coverage versus the binary one's ~80%). The real, robust win lives only in the coverage-independent metrics (AUROC, AUGRC).

**3. The finding that matters most: nothing *outside* the disagreement family worked.** Across the two searches I tested 40-plus *genuinely different* mechanisms — nearest-neighbour reachability, firing-set similarity to correct exemplars, density / typicality, conformal nonconformity, Bayesian weight uncertainty, causal do-interventions, dynamical-systems basin depth, spectral reconstructability, query-by-committee. **None beat the binary champion. Several fell below chance.** Plain confidence, temperature calibration, density, and attribution-mass all fail too. The load-bearing ingredient is specifically **two-view disagreement** — comparing the model to its *reliable-feature self*. Everything that works refines that; everything that abandons it loses.

**Updated bottom line:** the agreement *mechanism* is right, and surprisingly robust to a hard adversarial search. Grade it with reliable-code surprisal for a better trust *dial*. But the gain is in *ranking*, not fixed-coverage accuracy — and no non-disagreement signal I could invent does better. Scope unchanged: sentiment, one model, three OOD domains × two poolings (n = 6); reproduction in [`verify_surprisal.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/verify_surprisal.py) and [`coverage_matched_check.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/coverage_matched_check.py).
