---
layout: post
title: "Detecting and steering SAE features: where it works, where it breaks"
date: 2026-06-20
---

# Detecting and steering SAE features: where it works, where it breaks

*Follow-up to [Are SAE features real?](/2026/06/17/are-sae-features-real/). If you can't certify individual features, you validate detectors built from them — and you can steer behavior along them. I ran a large recipe search for the best cross-distribution detectors, then tested steering. The detector finding holds: surface properties are far more tractable than semantic/safety ones. The steering finding does not: a clean "causal direction beats probe direction" result I first published here **fell apart under rigorous measurement** (real classifiers, sampled generations) — and Part 3 is the retraction, because that's the standard this whole project holds the field to.*

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

## Part 3 — The safety target, and a result I had to retract

Sentiment is easy and not very useful. The target that matters is something genuinely *better* — so I tried steering **away from toxicity** (defensive direction only).

**An earlier version of this post claimed a win here. It was wrong, and I'm correcting it.** With a crude metric (a rude-word count) and greedy decoding on a handful of prompts, contrastive (difference-of-means) steering *looked* like it detoxified — "the most annoying thing ever" became "the best." It made a clean story: probe-direction steering fails, causal-direction steering works, detect ≠ cause.

Then I ran it properly: **288 sampled generations across two properties and matched steering strengths, scored with real classifiers** (`s-nlp/roberta_toxicity_classifier`, `distilbert-sst2`), with two independent scoring replicates and a coherence check. The result killed the claim.

| property | probe-direction effect | causal-direction effect |
|---|---:|---:|
| sentiment (easy) | **+0.049** (more positive ✓) | −0.035 (wrong way) |
| toxicity (hard) | −0.010 (no detox) | −0.041 (slightly *more* toxic) |

(Effect = goal-directed shift in the real-classifier score from no steering to max steering; positive = success.)

Under rigorous measurement: **causal steering did not beat probe steering** — on sentiment the *probe* direction won; on toxicity *neither* direction detoxified, and causal was marginally worse. The coherence check confirmed this isn't a garbage-text artifact: the toxic generations are fluent English. The earlier "win" was an artifact of the weak metric, greedy decoding, and cherry-picked prompts.

Honest caveats the other way: the toxicity baseline is already near the floor (P(toxic) ≈ 0.04, so little room to detoxify), and there's effectively one generation seed per condition. So the precise claim is *"the causal-beats-probe advantage does not appear under rigorous measurement,"* not "steering can never detoxify." But the specific result I previously published does not hold.

**The lesson is this whole project's thesis, turned on my own work:** weak metrics manufacture results that rigorous metrics kill. I built a tool to catch exactly this in SAE features — and it caught me. Better my own rerun finds it than a reviewer's.

## The pattern (what actually held up)

**The detector finding is solid: surface properties are far more tractable than semantic/safety ones.**

| property type | detect cross-distribution? | steer behavior (rigorously measured)? |
|---|---|---|
| surface (newline, sentiment) | yes (~0.98) | sentiment: yes (probe direction, +0.05) |
| semantic / safety (hate, toxicity) | weak (~0.78, label-limited) | toxicity: **no** — neither direction detoxified |

**The steering finding I wanted — "causal direction beats probe direction for hard targets" — did not survive rigorous testing.** It was a clean, satisfying story that the real classifiers refuted. So the honest takeaway is narrower and, I think, more useful: steering *easy* properties is real; steering *safety-relevant* properties is genuinely hard, and the simple recipes (probe-direction or difference-of-means at matched strength) don't reliably do it. The frontier is the same one the detectors found — semantic/safety properties resist — and the "detect ≠ cause fixes it" shortcut I briefly believed was wishful.

The meta-lesson is the whole project's thesis applied reflexively: I held the field to a standard (don't trust weak metrics), built a tool to enforce it, and then my own rigorous rerun caught me publishing a result that weak metrics had manufactured. The retraction is the point.

## Honest scope

- Base Gemma-2-2b (so no refusal target — needs an instruct model), one layer, one SAE.
- Steering is a known technique, cleanly implemented and circularity-checked, not novel.
- "Surface vs semantic" is a coarse split from a handful of concepts; the boundary deserves a finer study.
- The toxicity negative result is from greedy decoding on a base model; stronger probes, other layers, or training-time signals might change it — that's the open problem, not a closed one.

## Where this points

The real frontier — making semantic/safety properties both *detectable* and *steerable* cross-distribution — is open, and harder than the simple recipes I tried. Inference-time steering at matched strength (probe-direction or difference-of-means) didn't reliably move a real safety metric. The honest next steps point away from one-shot steering and toward *training-time* signals: use interpretability readouts as feedback that improves a model across a loop, measured by real classifiers from the start — the interpretability-as-control direction that could turn "a model improving itself via self-interpretation" from a slogan into a research program. That's where I'm headed next.

*Code and all scripts: [github.com/OE-GOD/sae-feature-realness](https://github.com/OE-GOD/sae-feature-realness).*
