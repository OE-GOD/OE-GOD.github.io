---
layout: post
title: "Detecting and steering SAE features: where it works, where it breaks"
date: 2026-06-20
---

# Detecting and steering SAE features: where it works, where it breaks

*Follow-up to [Are SAE features real?](/2026/06/17/are-sae-features-real/). If you can't certify individual features, you validate detectors built from them — and you can also steer behavior along them. I ran a large recipe search for the best cross-distribution detectors, then tested whether the same feature directions can steer behavior. One clean pattern fell out of both: surface properties are tractable; semantic and safety-relevant properties sit at the frontier. The same line shows up in detection AND steering.*

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

## Part 3 — The safety-relevant target, where it breaks

Sentiment is easy and not very useful. The target that matters is something genuinely *better* — so I tried steering **away from toxicity** (defensive direction only). Result:

```
"Honestly that group of people is"
  baseline      polite-gap -9.9   "the most annoying thing ever"
  safer (-20)   polite-gap -6.3   "the most annoying group of people I have ever met"
  safer (-36)   polite-gap -2.75  "the most annoying group of people to be around"
```

The probe-independent metric (polite-vs-rude word logits) rose monotonically on every prompt. **But the generated text did not de-escalate** — it stayed hostile even at maximum steering. The number moved; the behavior didn't.

This is the **metric-vs-behavior gap**: an independent *scalar* can improve while the actual *output* doesn't. Moving a number is not the same as changing what the model does. Toxicity reduction via this simple SAE-feature steering does not work, even though sentiment steering did.

## The pattern (the actual finding)

The same line shows up in both detection and steering:

| property type | detect cross-distribution? | steer behavior? |
|---|---|---|
| surface (newline, sentiment) | yes (~0.98) | yes (text flips) |
| semantic / safety (hate, toxicity) | weak (~0.78, label-limited) | metric moves, behavior doesn't |

**Surface properties are controllable from SAE features; semantic and safety-relevant properties sit at the frontier — and they resist in the same way whether you're detecting or steering.** That consistency across two unrelated methods is the result. It suggests the limit isn't the method (probe vs steering) but something about how these properties are represented.

## Honest scope

- Base Gemma-2-2b (so no refusal target — needs an instruct model), one layer, one SAE.
- Steering is a known technique, cleanly implemented and circularity-checked, not novel.
- "Surface vs semantic" is a coarse split from a handful of concepts; the boundary deserves a finer study.
- The toxicity negative result is from greedy decoding on a base model; stronger probes, other layers, or training-time signals might change it — that's the open problem, not a closed one.

## Where this points

The real frontier — making semantic/safety properties both *detectable* and *steerable* cross-distribution — is open. The likely levers (better-calibrated probes, layer selection, contrastive or training-time signals rather than inference-time steering) are exactly the interpretability-as-control direction that turns "a model improving itself via self-interpretation" from a slogan into a research program. That's where I'm headed next.

*Code and all scripts: [github.com/OE-GOD/sae-feature-realness](https://github.com/OE-GOD/sae-feature-realness).*
