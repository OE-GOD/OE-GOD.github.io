---
layout: post
title: "Compositional but not a direction: when the model's reasoning lives in attention, not a vector"
date: 2026-06-23
---

# Compositional but not a direction

*The capstone of the compositionality thread ([toolbox post](/2026/06/23/compositional-toolbox/)). We'd typed three composition operators as **directions** in the residual stream — negation (a shift), intensification (a gain), conjunction (averaging) — all readable and steerable. This post finds the operator type that **breaks the directions paradigm itself**: comparison. The model composes it correctly, but it is **not a vector you can read or steer** — it lives in attention. That's a real limit on the dominant interpretability toolkit, found from first principles.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b · behavioral + steering, reproduced*

---

## The first-principles prediction

Everything we'd done used *directions*: a concept is a direction, you read it by projection and write it by steering. But ask what "A is better than B" requires the model to represent: **which entity fills the winner role.** A direction is the *same vector added everywhere* — it is **position-agnostic.** It literally cannot say "the entity *at position i* is the winner."

So the prediction is sharp: if the model handles comparison at all, it **cannot** be a single residual direction. It must be **attention** — heads that move position-tagged information between the entities and the answer. Comparison should be **compositional but not a direction.**

## Step 1: comparison really does bind (not a position trick)

Same entities, same positions, flip only the comparator: "Toyota is **better** than Honda" vs "Toyota is **worse** than Honda." If the model tracks the *relation*, the winner flips; a position heuristic (always pick the first, or the most recent) can't flip.

It flips. Across 12 single-token pairs, the winner swings **+2.31** when you flip better↔worse, with a clean sign-flip on **67%** of pairs. (Honest wrinkle: there's a recency *bias* underneath — the baseline favors the second entity — so it's binding *modulated on top of* a position prior, not pure binding. But it is decisively **not** a pure heuristic.) The model genuinely assigns the comparison role.

## Step 2: but it's not a direction

Now build a comparison "role-direction" the *exact same way* the negation driver was built — `w = mean[ resid("A better than B") − resid("B better than A") ]` at the answer position, the "A-wins minus B-wins" vector. For negation this gave a clean, reusable, steerable lever (cosine 0.92, flips sentiment at k=2). For comparison:

| | negation | **comparison** |
|---|---:|---:|
| role-direction reusability (avg pairwise cosine) | 0.92 | **0.02** |
| steering flips the outcome? | yes (k≈2) | **no — 0% even at k=8** |

There is **no shared direction** — the per-pair "who wins" vectors point essentially at random (cosine 0.02). And steering with their average **never flips the winner** at any strength; it only nudges the magnitude. The recipe that worked perfectly for negation **completely fails** here.

## The result: the fourth operator type

> **Comparison is compositional but not a direction.** The model assigns the role correctly (it binds), yet that role assignment is **not** a vector you can read off or steer — it's carried by **attention**, moving position-tagged information the residual-direction toolkit can't capture.

That completes the operator typology:

| operator | type | lives in |
|---|---|---|
| negation | asymmetric shift | a **direction** (driver) |
| intensification | symmetric gain | a **direction** |
| conjunction | averaging | a **direction** |
| **comparison** | **role binding** | **attention** (not a direction) |

Three of the model's composition operators are directions you can read and steer. **The fourth isn't** — and we predicted exactly that from the geometry: a vector added everywhere can't encode *which* constituent plays *which* role.

## Why this matters for interpretability itself

The dominant interpretability paradigm — probes, SAE features, steering vectors, "concepts are directions" — is *built on the assumption that what the model computes with is a direction.* This result is a concrete case where **that assumption fails**: a genuine, behaviorally-correct piece of the model's reasoning that is **invisible to reading and immovable by steering**, because it's *structural* (attention routing), not *featural* (a direction).

So the honest scope of "concepts are directions" is: **great for properties (sentiment, truth, negation), blind to relations (who-did-what-to-whom).** The relational, position-dependent part of thinking — the part that makes "A beats B" different from "B beats A" — isn't in the directions at all. To see it you need **circuits** (attention heads, path patching), not vectors.

## Honest scope, and the next step

I showed the **direction-fails** half decisively (cosine 0.02, steering 0% at k=8) and the **binding** half behaviorally (flip +2.31). The complementary half — *localizing the attention heads that do the binding* (head ablation for necessity, position-swap patching for sufficiency) — I have **not** run; it's the natural next experiment and exactly the path-patching/IOI methodology this points toward. So the precise claim is "compositional but not a *steerable diff-of-means direction*," with attention-routing the first-principles inference. Also: 2B model, single-token entities, a recency bias in the behavior. Reproductions: [`comparison_gate.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/comparison_gate.py), [`comparison_direction_test.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/comparison_direction_test.py).

---

The compositionality arc ends where it had to: **some of how the model thinks is directions you can read and move — and some of it is wiring you can only see by tracing the circuit.** Comparison is the model reasoning in a register the directions can't reach.
