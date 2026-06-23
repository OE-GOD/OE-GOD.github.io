---
layout: post
title: "The driver map: finding what a model thinks with — and catching the tool when it goes blind"
date: 2026-06-23
---

# The driver map: finding what a model thinks with — and catching the tool when it goes blind

*The companion to [drivers and thermometers](/2026/06/22/drivers-and-thermometers/). That post showed the model **reasons with concepts but only carries facts** — read = write for a concept, read ≠ write for a fact. This one asks the engineering question: given a concept, which of its thousands of features does the model actually **think with**, and how do you find them at scale? The answer is a two-number map — and building it produced a small, perfect demonstration of why you can't trust the cheap tool: it went blind on a real driver, exactly where theory said it would.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b + Gemma Scope L12 SAE · sentiment, reproduced*

---

## The question, and why reading can't answer it

A concept like sentiment is spread across thousands of SAE features. Which ones is the model **using** to decide — the *drivers* — versus merely *carrying* — the *thermometers*?

You cannot read this off. A feature being **decodable** (you can predict sentiment from it) tells you it's *present*, not that any downstream circuit *consumes* it. The only thing that settles it is **causation**: change the feature and see if the answer moves. So every feature gets two scores:

- **read-score** — correlation with the concept (cheap: a dot product). *Finds candidates.*
- **cause-score** — does steering/ablating it move the output. *Finds the truth.*

**Driver = high cause. Thermometer = high read, low cause.**

## Two ways to measure cause — one honest, one cheap

**Honest (steering):** add the feature's direction to the residual stream and watch the sentiment output shift. A real, finite intervention — immune to the approximations below. Expensive (one pass per feature), so I ran it on the top-20 *readable* features.

**Cheap (attribution patching):** one backward pass gives a gradient-based cause estimate for **all 16k features at once** — the [Efficient AtP\*](/2026/06/03/efficient-atp-star-softmax/) idea. Scales, but it's a *local linear approximation*, and a methods review flagged that on a **residual-stream SAE at a mid-depth layer** it should produce **false negatives**: LayerNorm zeros the gradient along the stream and saturated nonlinearities read as zero slope — so a real driver can score ~0. (My own earlier result that [AtP degrades at depth](/2026/06/02/atp-degrades-with-depth/) is the same warning.)

So the plan was never "trust the cheap map." It was: build it, and **validate it against real steering.**

## The result: reading doesn't find the levers

Both methods agree on the headline:

- **AtP across 9,037 live features:** correlation between read-score and cause-score = **0.049** — essentially zero. Of the top-100 *readable* features, only **5** are in the top-100 by cause.
- **Steering on the top-20 readable features:** correlation **0.32**; about **5 of 20** are real drivers.

> **Most of a concept's readable features are thermometers — present, not used. Reading cannot tell you which ones the model thinks with; you have to measure cause.**

This is the same law the project opened with on Pythia (14 of 23 "monosemantic" features were thermometers), now on Gemma sentiment. The thing noticed at the very start turns out to be the spine of the whole arc.

## The tool went blind — exactly where theory said it would

Here's the part worth the whole post. Checking the cheap map against the four **steering-confirmed drivers**:

| driver | steering cause | AtP rank (of 9,037) |
|---|---:|---:|
| 8000 | 0.93 | **1** |
| 7234 | 0.51 | **3** |
| 14733 | 0.98 | **16** |
| **8836** | **0.55** | **1298** |

AtP caught three of four at the very top — and **missed 8836 completely**, a genuine driver (real steering moved the output 0.55) that the gradient buried at rank 1298. That is a textbook **false negative**: LayerNorm/saturation killed the gradient signal for a feature that *actually drives the answer.*

We watched our instrument mislead us — innocently, mechanically — and the *only* reason we caught it is that we'd poked 8836 for real. It also means AtP's "5% drivers" is a **lower bound**: the cheap map *undercounts* drivers because it goes blind on some. The faithful number (~25%, from steering) is higher.

## What this says about doing interpretability at all

Two layers of "the map is not the territory," and we hit both:

1. **The model's readable surface ≠ what it computes with** (thermometers exist — present but unused).
2. **Our cause-measuring tool ≠ the real causal effect** (the gradient goes blind under LayerNorm and saturation).

> To find what a model is *thinking with*, you must measure **causation, not correlation** — and even the causation tool has to be checked against a **real intervention**. Looking is not enough, at any level. You have to poke it and see what moves.

That's not paranoia; it's the lesson the data forced. The cheap map is genuinely useful — it ranked the strongest drivers first and is right at the extremes — but only *because* we know, from steering, where to trust it and where it lies.

## Honest method notes (for anyone rebuilding this)

The methods review named fixes I did **not** all apply here, so treat this as a prototype, not a finished atlas: use **integrated gradients** (not single-point grad×activation) to beat saturation; **stop-gradient through LayerNorm**; a **matched counterfactual** baseline (not zero-ablation); report **ranks, not magnitudes** (AtP inflates ~2×); and split cause into **necessity** (ablate) × **sufficiency** (steer), since one score mislabels redundant features. Positioned against Marks et al., *Sparse Feature Circuits* (2024), which builds causal SAE-feature circuits with the indirect-effect machinery this map is a concept-wide special case of. Scale: 2B. Reproductions: [`driver_detector.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/driver_detector.py) (steering), [`driver_map_atp.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/driver_map_atp.py) (AtP + the validation above).

---

The arc's question was "what is the model thinking?" The driver map is the operational answer: **of everything it represents, only a fraction is load-bearing — and finding which fraction means measuring what moves the answer, with a tool you've checked against the real thing.**
