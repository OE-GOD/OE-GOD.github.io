---
layout: post
title: "Removing spurious features makes a semantic detector generalize"
date: 2026-06-20
---

# Removing spurious features makes a semantic detector generalize

*Across [the](/2026/06/17/are-sae-features-real/) [first](/2026/06/20/detect-and-steer-frontier/) two posts, one wall kept showing up: semantic detectors built from SAE features don't generalize across distributions. A sentiment probe trained on movie reviews scores 0.66 F1 on product reviews. This post is a fix that survived adversarial verification — identify the features that encode the **domain** rather than the **concept**, drop them, and cross-distribution F1 rises to 0.74, with in-distribution performance unchanged. It beats a random-removal control by ~9σ, with a passing leakage audit. After [retracting a steering result last post](/2026/06/20/detect-and-steer-frontier/), I held this one to a high bar before believing it.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b + Gemma Scope SAE · sentiment, real labels*

---

## The problem: semantic signals don't transfer

A linear probe on SAE features detects sentiment well in-distribution (F1 ~0.87 on held-out SST-2) but drops to **0.66 on amazon_polarity** — a different domain (product vs movie reviews). The reason, from error analysis in the earlier work: the probe partly learns *domain vocabulary* ("film", "cinema", "plot") rather than *sentiment itself*. Those domain features are useless — worse, misleading — on a new distribution. This is the [spurious-correlation problem](https://arxiv.org/abs/2503.09532) (SAEBench calls it SCR), and it's the thing standing between SAE-feature detectors and a *trustworthy* semantic signal.

## The idea: drop the domain features, keep the concept features

SAE features are individually inspectable — so you can ask of each one: *does it discriminate the **domain** or the **concept**?* The recipe:

1. Train the sentiment probe on SST-2 (as usual).
2. **Identify domain-spurious features** using a domain classifier — train a logistic regression to tell SST-2 from rotten_tomatoes (a *different* domain, never amazon), and take its top-weighted features. These are the features that encode "which corpus is this," not "what sentiment."
3. **Remove them**, refit the sentiment probe on what's left.
4. Test on amazon — a domain used *nowhere* in selection.

This is interpretability-native debiasing: it works *because* you can name and remove specific spurious features, which raw activations don't let you do cleanly.

## The result

| | amazon F1 (cross-distribution) | in-distribution F1 |
|---|---:|---:|
| baseline (all features) | 0.661 | 0.874 |
| **de-spuriousing** (drop ~200 domain features) | **0.744** | 0.876 |
| random-removal control (same count) | 0.661 ± 0.008 | — |

Cross-distribution sentiment F1 rises **+0.083**, in-distribution performance is **preserved**, and it beats random removal of the same number of features by ~0.08. The effect is broad across removal counts (50–4000 all land +0.05 to +0.08), not a single lucky hyperparameter.

## The verification (the part that matters)

Last post I published a steering result that fell apart under rigorous measurement, and retracted it. So this time I ran the adversarial checks *before* believing the result — an independent re-implementation that imports none of the original code:

1. **Leakage audit — passed, with a negative control.** Amazon is never used to select features (verified by source inspection). To be sure the harness would *catch* leakage, I built a deliberately leaky selector that peeks at amazon — it scored **worse** (0.717 < 0.744). So the honest recipe isn't secretly exploiting an amazon-correlated signal; the gain is genuine domain-invariant denoising.
2. **Reproduced independently** — fresh float64 reimplementation matches the original numbers within 0.002.
3. **~9 standard deviations outside the random-control noise band** (3 seeds). Not regularization, not noise.

## Why it matters

This is the first clearly *positive, verified* result in this whole "are interpretability signals trustworthy?" arc — and it's a place where SAE features genuinely beat raw probes. You can't easily "remove the domain direction" from a raw activation probe; you can identify and drop specific interpretable features. It moves the central question forward: **a semantic signal you can trust across distributions** is exactly the prerequisite for any interpretability-driven control (or self-improvement) loop. Detection had to become trustworthy before control could.

## Honest scope

- One OOD target (amazon), one selection domain (rotten_tomatoes). The +0.08 generalizes across the rt-selected features but hasn't been shown to transfer to arbitrary unseen domains.
- The winning removal count (~200) was chosen by sweeping amazon F1, so the exact number is mildly optimistic — but the effect is broad, not a knife-edge.
- Sentiment has clean labels. The harder, noisy-label case (toxicity / hate) is untested here and is where the signal was weakest to begin with.
- This is interpretable feature selection for domain-invariance — related to SAEBench's SCR and the broader domain-generalization literature, executed on SAE features and verified end to end.
