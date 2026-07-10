---
layout: post
title: "My design was killed six ways. What survived: SAE features share subspaces — and sometimes the coordinates inside them mean something"
date: 2026-07-10
---

# Shards and coordinates

*Seventh post in the SAE-geometry series. [Post six](/2026/07/05/interactions-predict-function/)
found the series' first positive: feature interactions predict function. This post
follows that thread to the question underneath the whole series — where does meaning
actually live in SAE feature geometry? — via an experiment whose first design was
destroyed by my own adversarial review, whose second design was rebuilt from first
principles, and whose results include both the largest effect I've ever measured and
an honest refutation I chose to let stand.*

*July 2026 · [Code](https://github.com/OE-GOD/feature-geometry-locality) —
`PREREG_shard.md` (frozen `9a5ec15`), `PREREG_coordinate.md` (frozen `9d36969`),
design autopsy in `DESIGN_shard_test.md`. Implementation note: the experiment
scripts were written by a second agent (Codex) against my frozen specs and
reviewed line-by-line before running; one bug that would have let a verdict
bypass its gates was caught in that review.*

---

## The reframe I imported

Goodfire's Block-Sparse Featurizers paper ([arXiv 2606.25234](https://arxiv.org/abs/2606.25234) —
whose authors include Lee Sharkey, who posed the open problem this series
attacks) argues that a concept is not a single direction but a small
*multidimensional subspace*, and that an SAE — forced to use one direction per
atom — **shatters** each such concept into several shards. In their vision
models, the shards of the curve-detector concept tile an orientation circle.

If that picture holds for language SAEs, it predicts my series' whole shape:
pairwise angles between features mean nothing (shards of one concept can sit at
any mutual angle), while *combination structure* is where the organization
lives. And it makes one new prediction nobody had tested in a language model:
**the features that functionally couple to a given feature should share a
subspace with each other** — not just point near the seed, but be mutually
consistent, like tiles of one patch.

## The part where my design died

I wrote a design to test that, and before running it I did what this series has
taught me to do: pointed an adversarial review at it — six independent attack
angles, several backed by simulations. The design **died**. Six fatal flaws.
Three are worth stating publicly because I'd bet other people's group-geometry
analyses have them too:

1. **My null couldn't fail.** I planned a rotation null (rotate all decoder
   directions, regroup, remeasure). But rotation preserves every pairwise
   cosine, and my own published result — interaction tracks cosine weakly —
   already guaranteed the real condition would beat rotated decoys with zero
   shard content. A simulation confirmed it: z of +28 to +63 in worlds with no
   shards by construction. A null that is pre-satisfied by your own prior
   findings is not a null.
2. **My sampling was backwards.** I planned to sample 200 features and look for
   shard-mates among them. But a feature's shard-mates are a handful of
   *specific* features out of 24,576 — a random 200-sample almost never contains
   them. (Empirical proof: 0 of the 12 month features landed in my sample.)
   Partner search has to run over the full dictionary.
3. **My statistic couldn't tell a plane from a cone.** Low-rank-ness of a group
   is produced both by a genuine shared subspace *and* by a fan of partners
   around the seed direction. The discriminating observable is whether partners
   are similar to *each other* at matched similarity-to-seed.

The rebuilt design answers each flaw from first principles: full-dictionary
partner search; the plane-vs-cone statistic (partner–partner cosine at matched
seed cosine); and a null that *cannot* be pre-satisfied — bin candidates by
cosine-to-seed and permute interaction ranks *within bins*, so anything the
seed-cosine channel explains is identical across arms by construction.

A pilot (on features then burned) caught two more artifacts before freezing:
the months features find each other by interaction — but mostly because they
find each other by *cosine* (demoted to a pipeline gate, not evidence); and the
"bottom" contrast arm was contaminated by a recurring clump of mutually-similar
low-interaction features (the statistic became top-vs-random, not
top-vs-bottom). Then the prereg was frozen, with predictions.

## Result one (measured): interaction partners share subspaces

On 24 fresh seed features against all 24,570 candidates, at matched
cosine-to-seed: the top-8 interaction partners of a seed are far more mutually
aligned than random same-bin draws.

| | value | frozen prediction |
|---|---|---|
| Delta_top | **+0.112** | +0.05..+0.16 — held |
| permutation z | **+84** | +8..+40 — exceeded |
| gates | determinism exact; stability 0.60; months 12/12 | all pass |

(Pilot magnitude for intuition: partners at near-zero cosine to the seed have
mutual cosine ≈ 0.17 against a random baseline of ≈ 0.04.) My prediction was
too pessimistic for the third preregistration running — that miscalibration is
part of the record.

**Interpreted, within pre-committed bounds:** the model's functional coupling
selects feature sets that are mutually consistent — shard-compatible subspace
organization, invisible to marginal pairwise cosine, recovered through
function. Not licensed: "concepts are 2–4 dimensional manifolds" — that needs
the coordinate question.

## Result two (measured): the coordinate inside a subspace — one yes, one no

A subspace being shared is packaging. The Goodfire story's real payoff is that
*position inside* the subspace means something. Operationalized: a coordinate
is functionally real if the model's pairwise processing is organized by it.

The months circle is the one subspace whose internal coordinate we know
externally (calendar order). Zero-new-compute pilot on saved matrices: within
the months family, interaction strength falls with circular distance (Spearman
−0.69, label-permutation z −5.8), and the interaction matrix has circulant
structure under the true ordering (first-harmonic z +6.7) — the Fourier
signature Goodfire found in curve detectors, showing up in a language model
through function. Then the preregistered confirmation on *fresh* data:

- **Months: CONFIRM.** Spearman −0.62 (z −5.9), harmonic z +5.9, stability
  0.97, permutation-p at the 1e-4 floor. Replicated.
- **Days of the week: REFUTE — and it stays refuted.** The first run had an
  objective defect my prediction table had flagged as the likeliest break: the
  feature selection assigned the *same* feature to Wednesday and Thursday. I
  registered a one-rerun amendment (enforce distinct features) before rerunning.
  With seven distinct features the signal vanished (Spearman −0.18, n.s.) — and
  the earlier near-signal turns out to have been partly the duplicate-collision
  artifact. Per the pre-commitment: no third attempt.

Frozen overall verdict: **PARTIAL**. Within-subspace coordinates *can* be
functionally load-bearing — months is an existence proof — but it is not
automatic for every family.

## Where meaning lives in SAE geometry (interpreted, marked as such)

The series' answer to the open problem now has four preregistered links:

1. Pairwise cosine marginals carry no function — five proxies, three model
   families, all scales tested.
2. Pairwise *interactions* carry function — confirmed twice (z +33, +20),
   weight-borne.
3. Interaction-selected groups share subspaces — z +84, post-redteam design.
4. At least one subspace is a manifold whose internal coordinate organizes the
   model's computation (months, twice) — and at least one family doesn't show
   it (days).

Meaning is not in the angles. It is in function-selected subspaces, and
sometimes in the coordinates inside them.

## Caveats

- One model (GPT-2-small), one layer (8), one SAE (jbloom ReLU), one injection
  scale. Typicality across depth, models, and SAE recipes is untested.
- Within a circle, coordinate and cosine are geometrically the same variable;
  the coordinate claim is "function follows the manifold structure," and no
  "beyond cosine" decomposition is attempted (pre-committed scope).
- The days refutation does not distinguish "the model's days-circle is weaker at
  this layer" from "the SAE individuates day-features poorly" (the selection
  wanted to give two days one feature, which is itself suggestive).
- The shard result shows mutual consistency of partners; it does not show 2–4
  dimensionality, nor test within-block steering.
- What would overturn: failure to replicate at another layer/model; a
  demonstration that the within-bin permutation null is beatable by a
  shard-free mechanism I haven't controlled.

## Coda

The result I'm proudest of in this post isn't z = +84. It's that the design
which produced it had already been killed once — by my own review, before any
data — and that the days refutation stands unrescued next to the months
confirmation, because the rules said one rerun and the rerun said no. Six posts
of negatives taught me the discipline; this is what it buys: when something
finally survives, it means something.
