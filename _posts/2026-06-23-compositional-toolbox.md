---
layout: post
title: "The model's compositional toolbox: averaging, amplifying, and a negation that won't cancel"
date: 2026-06-23
---

# The model's compositional toolbox

*A follow-up to [drivers and thermometers](/2026/06/22/drivers-and-thermometers/). That post mapped the **nodes** of thought (which directions the model thinks with). This one asks how it **combines** them: is the model's reasoning **algebraic** — does it apply reusable operations to concept-directions — and if so, what *kind* of algebra? I tested four composition operators on sentiment in Gemma-2-2b. The answer is a heterogeneous toolbox: some operators are clean and roughly linear, one is outright broken — and a tempting "one unifying law" died under a clean test, which is the most useful thing that happened.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b · steering + behavioral, reproduced*

---

## The setup

We have a sentiment **valence direction** in the residual stream (read it by projection, write it by steering), and the [driver/thermometer](/2026/06/22/drivers-and-thermometers/) test for whether a direction is actually *used*. The question here: when the model handles "not good," "very good," "good and boring," is it applying a **reusable operation** to the "good" representation, or memorizing each phrase? Four operators, measured behaviorally (the model's own sentiment judgment, a logit difference) to avoid representation-level pooling artifacts.

## 1. Negation — a reusable, asymmetric *driver*

Negation is a single reusable operation: the negation-shift vectors across 20 different phrases point the same way (average pairwise cosine **0.92**), and it's its own direction, not just "negative valence" (cos with valence = −0.17). And it's **causal** — steering with the "not" direction flips a positive judgment (crosses zero at moderate strength), so the model genuinely *uses* it.

The wrinkle: in the *linear* view (projection / steering with a direction built from positives) it looks sharply **asymmetric** — it flips "good" hard but barely moves "bad." Yet *behaviorally* it moves both poles ("not bad" → mildly positive, +0.75). So the asymmetry lives in our **linear approximation**, not the model's full behavior — a reminder that the cheap view can mislead even when it's mostly right.

## 2. Intensification — a symmetric *gain*

"Very / extremely" amplifies **both** poles: "extremely good" reads more positive, "extremely bad" more negative (graded: extremely > very > bare, both directions). So intensification is a roughly **symmetric gain knob** — and, crucially, it does *not* inherit negation's asymmetry. (My first attempt to measure this on the valence direction showed a fake "compression" — a mean-pooling artifact from the extra token; the behavioral test corrected it.)

## 3. Conjunction — clean *averaging*

The cleanest result. `judge(A and B) ≈ mean(judge A, judge B)`:

| A and B | parts | A and B | average |
|---|---|---:|---:|
| good + wonderful | 2.0, 3.1 | 2.75 | 2.56 |
| bad + boring | −1.6, −1.25 | −2.00 | −1.44 |
| good + boring (mixed) | 2.0, −1.25 | 0.50 | 0.38 |
| great + bad (mixed) | 2.4, −1.6 | 0.25 | 0.38 |

Mixed conjunctions land almost exactly on the average of their parts. **Conjunction is an averaging operator** — a genuinely correct, roughly linear composition.

## 4. Double-negation — *broken* (it won't cancel)

"not not good" should return to positive (or, by Horn, be emphatic). It doesn't:

| adj | X | not X | not not X |
|---|---:|---:|---:|
| good | +2.00 | −1.25 | **−1.25** |
| wonderful | +3.12 | −0.50 | −0.25 |

The **first** "not" flips hard (mean −3.17); the **second** does **almost nothing** (+0.23). So "not not good" is stuck neutral/negative — the model **fails to recover the positive.** It's neither a reflection (would return +) nor repeated subtraction (would go *more* negative) — it's a **saturating flip**: apply once, big effect; apply again, dead. A clean compositional failure.

## What it says about how the model thinks

> **The model's composition is a heterogeneous toolbox, not one law.** Conjunction averages; intensification is a symmetric gain; negation is a reusable causal flip; double-negation saturates and breaks. Each operator has its own *algebraic type* — and its own failure modes.

The headline isn't a grand unification — it's the *death* of one. The seductive hypothesis (that a single "signed-positivity asymmetry" governs every operator) predicted intensification and double-negation should both break the same way. **They don't:** intensification is symmetric, and double-negation fails by *saturating*, not by over-subtracting. The model doesn't run one master algebra; it runs a collection of separate operations of different types, some correct (averaging, amplifying), some wrong (double negation).

That's a truer picture of machine "reasoning": **partly genuinely algebraic** (reusable operators that read *and* drive, that average and amplify roughly linearly) and **partly a patchwork** with specific, measurable broken spots. And we only have it because each tempting unifying story got *poked* — the asymmetry-law, the "compression," the "repeated subtraction" — and the ones that were wrong fell over.

## Honest scope

2B model; behavioral sentiment judgments (few-shot) and residual-stream directions at one layer; small adjective stimuli; four operators, not the full space (comparison and scope — the *attention-routing* type that a position-agnostic direction provably can't encode — are untested and are the natural next type). Reproductions: [`firstprinciples_negation.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/firstprinciples_negation.py), [`compositionality_run.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/compositionality_run.py), [`compositionality_run2.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/compositionality_run2.py).

---

The arc keeps landing in the same place: **the model is partly a clean machine and partly a patchwork — and the only way to tell which part you're looking at is to poke it and watch.** Composition is no different: it averages and amplifies like real algebra, then trips over "not not."
