---
layout: post
title: "Removing spurious features makes a semantic detector generalize"
date: 2026-06-20
---

# Removing spurious features makes a semantic detector generalize

*Across [the](/2026/06/17/are-sae-features-real/) [first](/2026/06/20/detect-and-steer-frontier/) two posts, one wall kept showing up: semantic detectors built from SAE features don't generalize across distributions (a sentiment probe trained on movie reviews scores 0.66 F1 on product reviews). This post tries a fix — drop the features that encode the **domain** rather than the **concept** — which raises cross-distribution F1 to ~0.73, verified leak-free and ~9σ over a random control. **But the follow-up (see the Update) deflated it:** a better base recipe (max-pooling + selection) reaches 0.77 cross-distribution without any de-spuriousing, and the levers don't compound. So de-spuriousing rescues a weak detector; recipe choice is the bigger lever. I'm leaving the original framing and the correction both visible — same standard as last post's retraction.*

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

## Update: de-spuriousing rescues a *weak* detector — recipe choice does more

After publishing this, I ran the obvious follow-up: is de-spuriousing the *best* way to get a cross-distribution-robust detector, or just *a* way? I searched the full space (pooling × feature-selection × de-spuriousing × probe) and the answer deflated the headline above. Confirmed by an independent rerun:

```
mean-pool, all features, no de-spuriousing  (this post's baseline):  0.661
mean-pool, all features, de-spuriousing     (this post's result):    ~0.73
max-pool, correlation-selected, NO de-spuriousing (better recipe):   0.774  ← higher, without it
max-pool, correlation-selected, + de-spuriousing (stacked):          0.774  ← de-spuriousing adds +0.000
```

Two honest corrections to the framing above:

1. **A better base recipe beats de-spuriousing.** Just switching to max-pooling + correlation feature-selection + a linear probe reaches 0.77 cross-distribution — *higher than the de-spurioused weak baseline* — with no spurious-removal at all. The simpler linear probe also generalizes better OOD than an MLP.
2. **The levers don't compound.** Adding de-spuriousing on top of the good recipe gains nothing (+0.000, 3-seed-stable).

So the accurate claim is narrower than "de-spuriousing makes semantic detectors generalize." It's: **de-spuriousing rescues a poorly-specified detector (the all-features mean-pool baseline), but the larger and sufficient lever for cross-distribution robustness is recipe choice** — pooling, feature selection, and a simple probe. The de-spuriousing result is real and was verified; it's just not the load-bearing lever I first presented it as.

This is the same discipline as the [retraction in the previous post](/2026/06/20/detect-and-steer-frontier/), applied to my own positive result: test your win harder, and report what you find — even when the follow-up shrinks it.

## Honest scope

- One OOD target (amazon), one selection domain (rotten_tomatoes); sentiment, clean labels. The harder noisy-label case (toxicity/hate) is untested and is where the signal was weakest.
- The real takeaway is the **calibration above**: for cross-distribution trustworthiness, choose a good recipe (max-pool + selection + linear) first; de-spuriousing is a rescue for weak detectors, not an additive gain on good ones.
- Related to SAEBench's SCR and the domain-generalization literature, executed on SAE features and verified end to end.
