---
layout: post
title: "After five negatives, a preregistered positive: feature interactions predict function, and the weights carry it"
date: 2026-07-05
---

# The first positive

*Sixth post in the SAE-geometry series. Posts one through five killed every claim
that the arrangement of SAE features means something: raw geometry (five ways), at
scale (three families), and in the model's own coordinates (preregistered,
instrument-limited). This post reports the series' first CONFIRM — preregistered,
gate-passed, control-checked — and states exactly how much it does and doesn't
claim.*

*July 2026 · [Code](https://github.com/OE-GOD/feature-geometry-locality) —
`PREREG_interaction_cofire.md`; registration commit `2abd7a0`, result `36746b4`,
control `3b00cef`*

---

## The one signal that kept not dying

Back in [part two](/2026/07/02/sae-geometry-part-two-interactions/), one measurement
survived the spectrum-matched null that killed everything else: **interactions**.
For a pair of features, inject both decoder directions into the residual stream and
ask whether the MLP's response is more than the sum of its responses to each alone:

```
I(i,j) = ‖mlp(h0 + di + dj) − mlp(h0 + di) − mlp(h0 + dj) + mlp(h0)‖
```

That's second-order structure — not where two features *point*, but what happens
when they're *combined*. At the time it correlated with geometry at a modest +0.16
and I honestly filed it as "real but thin."

What nobody had ever computed — not in this project, not anywhere I know of — was
whether this interaction structure predicts *function*: which features actually
fire together on real text.

## Instrument first, hypothesis second

The ruler test (post five) taught me the order of operations the hard way: check
that your instrument measures a stable thing *before* designing an experiment on
it. So, pilot first: compute the full 19,900-pair interaction matrix on two
disjoint halves of the data and correlate. Result: **+0.95 stability beyond the
known scalar axis** (the ruler's responses managed 0.26). A real coin.

Only then did I preregister the hypothesis — on a *fresh* feature sample, disjoint
from the pilot's, with the statistic, the rotation null, the gates, the
confirm/refute lines, and my honest predictions frozen in a public commit
(`2abd7a0`) before the number existed. The null is the same blade that killed the
whole series: rigidly rotate all 200 decoder directions, which preserves every
pairwise angle exactly and destroys only their alignment to the model's weights.
Whatever survives that comparison is carried by the alignment — the thing the
claim is actually about.

## What happened (measured)

Gates: bit-exact determinism; split-half stability +0.948 on the fresh features.

| | rho(interactions → co-firing) | partial, controlling raw cosine |
|---|---|---|
| real decoder | **+0.113** | **+0.108** |
| 8 rotation nulls | +0.012 ± 0.003 | +0.011 ± 0.003 |
| **z** | **+33.5** | **+32.4** |

The frozen rule said CONFIRM needs z ≥ +4 on both channels. Both came in above
+32. First confirm of the series.

Prediction scorecard, misses first: my frozen ranges were rho +0.00..+0.06 and z
+0..+4 — **too pessimistic, by a lot**. After five negatives I had stopped
believing positives were available. The register is predict-then-test precisely so
that this kind of miscalibration is on the record too.

## The control that earned the interpretation

One boring channel could have produced the confirm. The interaction is measured at
base points sampled from real text — and rotated decoy features never *occur* in
real text. So maybe co-firing pairs score high interactions simply because both
features are sometimes already sitting in the base point (an occupancy echo), not
because the weights couple them.

The control: recompute everything at base points where **none of the 200 features
fires** — residuals the features are verifiably absent from. If the signal is an
occupancy echo, it dies there.

It didn't move: rho +0.111 (vs +0.113), z = +26.5. The coupling is carried by the
MLP's weights — visible in how the network's fixed machinery responds to direction
pairs, even at points where the features themselves are absent.

## What this does and doesn't claim

**Measured:** in GPT-2-small at layer 8, pair-specific interaction structure
(scalar axis removed) predicts co-firing on held-out text at rho ≈ 0.11, roughly
10× its perfectly matched null, robust to controlling raw cosine and to removing
the occupancy channel.

**Interpreted (my reading):** the network's weights encode which feature pairs are
co-functional, and that organization is readable in second-order response
curvature. The organization of SAE features exists — it just doesn't live where
everyone looked. Not in the angles between features (dead, five ways, three
families). Not in first-order responses (unstably encoded at best). In the
*combinations*.

**Caveats, the usual register:** one model, one layer, 200 features, one function
measure. Effect size: rho 0.11 is ~1.3% of pair-rank variance — a genuine
organizational signal, not yet a usable map. The direction of explanation is open:
the weights may couple these pairs *because* they co-occur in training, or the
co-occurrence may exploit the coupling; nothing here separates those. What would
overturn it: failure to replicate on a fresh sample, another layer, or another
model — all cheap tests, none run yet.

## Coda

Five posts of negatives were the price of being able to say this one carefully.
The same nulls that killed my halos, my causal z=200, my months-circle, and my
scale story are the reason a +0.11 with z=+33 means something: it beat the decoys
that everything else lost to, under rules frozen before the data existed, with the
one boring explanation controlled away.

The blank map was never the end of the story. It was the elimination step. What's
left standing, after everything eliminable was eliminated, is a small, stable,
weight-borne signal saying the features are organized by how they *combine*. Next:
whether that interaction graph has communities and hierarchy — whether the
organization has *shape*.
