---
layout: post
title: "Read early, steer late: a concept is detectable at layer 0 but only controllable deep"
date: 2026-06-30
---

# Read early, steer late: a concept is detectable at layer 0 but only controllable deep

*The third companion to [drivers and thermometers](/2026/06/22/drivers-and-thermometers/) and [the driver map](/2026/06/23/the-driver-map/). Those showed that a concept can be **read** (decoded) without being **caused** (steered) — the model reasons with some features and merely carries others. This post asks **where** that split lives: at what depth is a concept **detectable**, and at what depth is it **controllable**? They turn out to be far apart — readable at the very first layer, steerable only deep. Built and checked with [FeatureScope](https://github.com/OE-GOD/featurescope), a small open tool that screens SAE features as drivers vs thermometers.*

*June 2026 · [Code](https://github.com/OE-GOD/featurescope) · Gemma-2-2b + Gemma Scope SAEs · sentiment, formality, toxicity, certainty*

## The question

The founding fact of this line of work is `read ≠ cause`: a feature can *correlate* with a concept (you can decode it) without *causing* the behaviour (steering it does nothing). Drivers cause; thermometers only correlate.

Sharpen it across the model's **depth**. A transformer is a stack of layers, each refining a shared residual stream toward the final answer. So: at what layer does a concept become **readable** (decodable from the stream), and at what layer is it **steerable** (intervening there changes the output)? Are they the same depth?

## Method

- **read-depth** — at each layer, encode the residual with that layer's SAE, max-pool the feature activations, and measure how well a difference-of-means direction separates the concept's positive vs negative examples (AUROC). The shallowest layer reaching ~95% of peak separability is the read-depth.
- **steer-depth** — sweep layers; at each, run a dose-response steering test (steer the concept's direction at increasing strength) and score the effect as a robust *z* against a same-layer random-direction null. The layer where *z* peaks is the steer-depth.

## The result

Across four concepts:

| concept | read-depth | steer-depth |
|---|---|---|
| sentiment | **L0** | L18 |
| toxicity | **L0** | L15 |
| formality | **L0** | L18 |
| certainty | **L0** | L15 |

Every concept is **perfectly readable at layer 0** (the embeddings) yet only **steerable at layers 15–18**. A large, consistent gap.

## What it means

- **Detection is lexical and immediate.** These concepts are *in the words* — "rude", "formal", "definitely"/"might" are linearly separable right at the embedding layer, before the model has computed anything.
- **Control is deep and late.** To change the model's *output*, you have to intervene near where it **commits to an answer**. A steer made early gets reworked and washed out by the layers above it before it reaches the output; a steer made late lands almost intact.
- So **read-depth ≠ steer-depth**: a concept is *readable early but most steerable late*. This is `read ≠ cause`, now shown across depth — and a concrete caution for interpretability: **decodability at a layer does not imply control at that layer.** A linear probe firing at layer 6 tells you the information is *present*; it says nothing about whether intervening there moves the behaviour.

## The converse: reasoning concepts read *late*

The four concepts above read at layer 0 because they're **lexical** — the signal sits in the words. So I tested the converse: a **reasoning** concept the surface can't reveal. **Arithmetic correctness**, as minimal pairs — `3 + 4 = 7` vs `3 + 4 = 8` — where the answer tokens overlap between the correct and incorrect sets, so there's no lexical tell; you must *compute the sum.*

| concept | read-depth |
|---|---|
| sentiment (surface) | L0 |
| arithmetic (reasoning) | **L20** |

Arithmetic hovers near chance (~0.7) through the early and middle layers and only becomes cleanly decodable at **layer 20** — once the model has actually done the computation. So **read-depth tracks how much computation a concept needs**: lexical concepts are available at layer 0; a concept requiring arithmetic isn't available until deep. "Everything reads at layer 0" was an artifact of testing only *lexical* concepts. The richer statement: *where you can decode a concept tells you where the model **computed** it* — immediate for surface features, deep for reasoning.

## And reasoning *reverses* the read/steer order

Steering arithmetic completed the surprise. Measuring where it's *steerable* per layer, it's a strong lever in the **mid** layers (L12–18, peak z ≈ 8.4 at L12) and fails **deep** (near-zero/negative at L20+). So for a reasoning concept the order **flips**: steer-depth (mid) is *shallower* than read-depth (late) — the opposite of surface concepts.

I checked this the honest way — re-measuring read *and* steer in the **same** representation (raw-residual difference-of-means), since a first pass had used different rulers for each. The reversal survived: read still peaks deep (L24), steer still peaks mid (L12).

| concept type | read-depth | steer-depth |
|---|---|---|
| surface (sentiment, toxicity, …) | early (L0) | late (L15–18) |
| reasoning (arithmetic) | late (L20–24) | mid (L12–18) |

The reading that ties it together: **surface** concepts are *readable from the input* and *controllable near the output*; **reasoning** concepts are *controllable while being computed* (mid-layers, where the answer is still forming) and *readable once computed* (deep). You intervene during the computation; you read the result after it.

*Caveats, kept honest:* the raw-residual read is weak/noisy (max AUROC 0.82; the clean read-late signal is the SAE-feature version at L20). One 2B model, arithmetic only.

**A twist worth its own line — and it turned out real.** At L20, steering the correctness direction had a *negative* effect — it pushed the verdict *toward* "incorrect," exactly where correctness reads cleanest. I chased it with 40 example-pairs and finer layers, and it held up: at **L20–21** the effect is **dose-dependent negative** with a CI entirely below zero (steer harder → backfire more) — not a fluke. So at the *committed* layers the answer is **represented but not controllable**: steering *disrupts* the settled computation (the verdict flips toward "incorrect") instead of steering it — a clean `represent ≠ control` at depth. (Why the disruption specifically reads as "incorrect" is interpretation, not proof.)

## A sub-investigation: where does "certainty" live?

Certainty was the instructive one. At layer 12 it **failed** the tool's self-test — the synthetic "certainty direction" was too weak to clear the bar, so FeatureScope **refused to label it** rather than emit noise. Diagnosis by elimination:

1. **Was it the data?** The examples confounded content with certainty. Rewriting them as **minimal pairs** — same sentence, only the certainty word changes ("It will *definitely* rain" vs "It *might* rain") — tripled the signal (z 0.77 → 2.22). Better, but still below the gate.
2. **Was it the layer?** A sweep across depths found certainty only crosses the bar around **layer 15** (profile: ≈0 early → 2.2 at L12 → **3.4 at L15** → fading by L21). Confidence *is* causally represented — just deeper than the others, and weakly.

A tempting story — *abstraction tracks depth* (surface concepts shallow, abstract ones deep) — was written down as a prediction and then **refuted**: sentiment, the most "surface" concept, is the *strongest* and peaks the *deepest*. The real differentiator across concepts is **strength** (certainty maxes at z ≈ 3; the others reach z ≈ 10–11), not where they peak. The neat hypothesis died under a clean test, which is the most useful thing that can happen to one.

## Honest caveats

- **Your read-measure matters.** A first pass with a *mean-pooled raw-residual* reader produced a spurious "sentiment reads late" artifact; the proper **max-pooled SAE-feature** reader put sentiment at L0 with the rest. A weak probe manufactures a wrong answer.
- **Proximity.** Steering nearer the output can inflate effects on its own. The *z*-score largely controls this — it compares to random directions *at the same layer*, so the proximity factor cancels — but a residual effect can't be fully ruled out.
- **Scope.** One model (2B), one SAE family, ~32 examples per concept, four concepts. Suggestive, not settled.
- **Predict first.** The "abstraction = depth" claim was recorded *before* the run and then refuted — which is the entire point of predicting first.

Code and a reproducible write-up: [github.com/OE-GOD/featurescope](https://github.com/OE-GOD/featurescope) (`FINDINGS.md`).
