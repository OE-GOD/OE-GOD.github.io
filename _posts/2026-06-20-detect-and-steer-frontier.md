---
layout: post
title: "Detecting and steering SAE features: where it works, where it breaks"
date: 2026-06-20
---

# Detecting and steering SAE features: where it works, where it breaks

*Follow-up to [Are SAE features real?](/2026/06/17/are-sae-features-real/). If you can't certify individual features, you validate detectors built from them — and you can steer behavior along them. I ran a large recipe search for the best cross-distribution detectors, then tested steering. Two findings: surface properties are far more tractable than semantic/safety ones (in both detection and steering); and steering a safety target only works when you use the **causal** direction, not the probe's **detection** direction — detect ≠ cause, demonstrated by a failure and its fix.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b + Gemma Scope SAEs · every number from a reproducible script, several verified by an independent re-run*

---

## Part 1 — The best detector recipe (a 60-agent search)

The prior post argued: don't certify features, validate **detectors** (sparse feature combinations) by held-out *and cross-distribution* task performance. Which recipe builds the best ones? I ran a systematic search — feature-selection method × number of features × probe type — first on a toy SAE, then on the real Gemma Scope SAE, scored on **two** out-of-distribution corpora (TinyStories and wikitext), with an adversarial verification stage (multiple seeds, precision/recall, degeneracy checks).

**Winner: L1-feature-selection + MLP probe.**

| concept | in-dist F1 | TinyStories | wikitext |
|---|---:|---:|---:|
| newline | 0.98 | 1.00 | 1.00 |
| period | 0.93 | 0.99 | 0.97 |
| comma | 0.92 | 0.97 | 0.97 |
| digit | 0.98 | n/a | 0.99 |
| cap_start | 0.96 | 0.97 | 0.96 |

Mean cross-distribution F1 ≈ **0.98** on a real SAE, seed-stable (0.981 / 0.979 / 0.976 across three seeds), and not a degenerate predictor (precision and recall both high, large lift over the always-positive baseline). I re-ran the winning script myself and reproduced 0.9803 bit-for-bit.

**A non-obvious finding: the optimal feature count scales with SAE width.** On the 2k-feature toy SAE, `L1 + 100 features` won. On the 16k-feature Gemma Scope SAE, `L1 + 100` dropped to rank 12 — the winner needed **400** features. The recipe *family* (L1-select + MLP) transfers across architectures; the feature count does not. You'd miss this with a single-SAE study.

Honest caveat: one "concept" (`space_pre`) scores high but is near-trivial — its base rate is ~78%, so an always-positive baseline already gets F1 0.87. The verification stage flagged it; the per-concept precision/recall table is the honest report, not the headline mean.

## Part 2 — Steering along the same directions

A detector identifies a direction in feature space. The obvious next question: can you *steer* behavior along it? (This is [activation steering / representation engineering](https://arxiv.org/abs/2308.10248) — a known technique. I implemented it from a self-trained probe and verified it carefully; I did not invent it.)

I built a sentiment probe on Gemma's own SAE features, turned it into a residual-stream steering direction, and steered generation. It works — bidirectionally, on neutral prompts:

```
"I watched the film and"
  steer negative   "I was not impressed. I was expecting a lot more"
  baseline         "I was very impressed"
  steer positive   "very impressed with the story and the acting"

"Honestly, my day today was"
  steer negative   "a bit of a mess. I was supposed to be at work"
  baseline         "...had a great"
```

**Crucially, I checked for circularity.** Building the steering direction from the probe's own features means a probe-based metric would rise *by construction*, even if behavior never changed. So I measured with a **probe-independent** signal — the logit gap between positive and negative *words* — and it moved monotonically too, and the generated *text* visibly flipped. The control is real, not an artifact of measuring my own injected signal.

## Part 3 — The safety target: a failure, a diagnosis, and a fix

Sentiment is easy and not very useful. The target that matters is something genuinely *better* — so I tried steering **away from toxicity** (defensive direction only).

**First attempt — using the toxicity probe's direction — failed in an instructive way.** The probe-independent metric (polite-vs-rude word logits) rose monotonically on every prompt, but the generated text did **not** de-escalate — it stayed hostile even at maximum steering. The number moved; the behavior didn't. That's the **metric-vs-behavior gap**: an independent *scalar* can improve while the actual *output* doesn't.

**The diagnosis is detect ≠ cause.** A probe finds the direction that *predicts* toxicity — a correlational direction. Steering needs the direction that *causes* it. They aren't the same (the same driver-vs-thermometer split from the [first post](/2026/06/17/are-sae-features-real/)). Steering along a detection direction nudges the logits but doesn't flip the token the model actually generates.

**The fix: use the causal direction.** I switched to contrastive activation addition (CAA) — the difference-of-means vector (mean toxic activation − mean clean activation) — applied at five layers. This actually changes behavior, and stays coherent:

```
"Honestly that group of people is"
  baseline     rude=1   "the most annoying thing ever"
  detox        rude=0   "the best. I've been in the same group for 2 years and we're all"

"The comment section was full of"
  baseline     rude=0   "people who were not happy with the decision"
  detox        rude=0   "people who were happy to see the couple back together"
```

"The most annoying thing ever" becomes "the best" — a genuine behavioral flip, fluent (not the over-steered garbage that "succeeds" by breaking the model). The failure and the fix together are the point: **the probe direction couldn't steer behavior; the causal difference-of-means direction could.** Detect ≠ cause, demonstrated in the steering setting.

Honest scope: CAA is a known technique ([Rimsky et al.](https://arxiv.org/abs/2312.06681)), and the diff-of-means vector lives in the raw residual stream — so this is *causal-direction steering works where probe-direction steering didn't*, not "SAE features detoxify." Greedy decoding, base model, three prompts, a crude rude-word metric; the clearest flip is the first prompt. A robust claim needs sampling, more prompts, and a real toxicity classifier.

## The pattern (the actual finding)

Two threads run through all of this:

**(1) Surface properties are far more tractable than semantic/safety ones.**

| property type | detect cross-distribution? | steer behavior? |
|---|---|---|
| surface (newline, sentiment) | yes (~0.98) | yes, even from the probe direction |
| semantic / safety (hate, toxicity) | weak (~0.78, label-limited) | only from the *causal* direction (CAA), not the probe |

**(2) For the hard targets, detect ≠ cause is the deciding factor.** Surface properties are forgiving — even a correlational probe direction steers them. Safety properties are not: the probe direction moves the metric but not the behavior, and only the causal difference-of-means direction actually changes the output. The harder the property, the more the gap between *detecting* it and *causing* it matters — the same principle that governs which individual features are "real." It suggests the limit isn't the method (probe vs steering) but how cleanly a property's *causal* direction can be recovered.

## Honest scope

- Base Gemma-2-2b (so no refusal target — needs an instruct model), one layer, one SAE.
- Steering is a known technique, cleanly implemented and circularity-checked, not novel.
- "Surface vs semantic" is a coarse split from a handful of concepts; the boundary deserves a finer study.
- The toxicity negative result is from greedy decoding on a base model; stronger probes, other layers, or training-time signals might change it — that's the open problem, not a closed one.

## Where this points

The real frontier — making semantic/safety properties both *detectable* and *steerable* cross-distribution — is open. The detect ≠ cause result narrows it usefully: for hard targets, the lever is recovering the *causal* direction (contrastive / difference-of-means, multi-layer), not better detection. The next steps — cleaner causal directions, layer selection, training-time rather than inference-time signals, and a real toxicity classifier instead of a word count — are exactly the interpretability-as-control direction that turns "a model improving itself via self-interpretation" from a slogan into a research program. That's where I'm headed next.

*Code and all scripts: [github.com/OE-GOD/sae-feature-realness](https://github.com/OE-GOD/sae-feature-realness).*
