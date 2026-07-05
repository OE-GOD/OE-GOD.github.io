---
layout: post
title: "The map was mostly blank: why SAE feature geometry doesn't predict function"
date: 2026-07-03
---

# The map was mostly blank

> **Correction, July 5, 2026.** The central *mechanism* this post proposes — that the
> map is blank *because* the model's response is ~99% feature-agnostic — did not
> survive a scale-and-recipe test. That ~99% turned out to be a property of the SAE
> training recipe, not of the network. The negative result itself (geometry does not
> predict function) survived and got stronger. Details in the dated correction at the
> end of this post and in the [follow-up post](/2026/07/05/sae-geometry-at-scale/).
> The original text below is unedited.

*Third and last in a series. [Part one](/2026/07/02/sae-geometry-does-not-predict-function/)
killed five ways of showing SAE feature geometry reflects function — all
negative, all spectrum artifacts. [Part two](/2026/07/02/sae-geometry-part-two-interactions/)
found the one signal that survived a real null (feature interactions) and showed
it was thin and mostly global. This post asks the only question left: not whether
geometry predicts function, but **why it can't** — and why that's a better result
than the positive I was hoping for.*

*July 2026 · [Code](https://github.com/OE-GOD/feature-geometry-locality) ·
a first-principles post-mortem on a project of negative results*

---

## The setup

Two posts, one shape: I kept finding beautiful results and killing them. A halo
where close features looked functionally similar — a random matrix reproduced it.
A causal signal at z ≈ 200 — the right null dropped it to 0.33. SAE features for
the months landing in a circle — the SAE was just inheriting the model's geometry,
adding nothing. A local interaction signal — real, but +0.16 and beaten 2-to-1 by
a number that uses no geometry at all.

Five deaths in a row invites a lazy conclusion: "there was nothing there." That's
not what happened. Something specific was going on, and it took inverting the
question to see it.

## The causal chain nobody states out loud

Write down what these quantities actually are.

A feature's **geometry** is its decoder direction — the vector it writes into the
residual stream. Its **function** is what the model *does* with that write:
downstream attention and MLPs process it into a change in behavior. So function
isn't independent of geometry. Function is `model(geometry)` — the geometry is the
input, and the model is the fixed function that turns it into an effect.

In principle, then, geometry should determine function completely. So why couldn't
I predict one from the other?

## What every experiment was actually saying

Because of a fact that showed up, identically, in every proxy I tried: **when you
perturb a feature, the model's response is ~99% one shared mode that is the same
no matter which feature you touched.**

Ablate any feature and the output distribution collapses toward the same set of
glitch tokens (a generic "I've been damaged" response). Measure co-firing and 99%
is a single activity-level axis. Measure interactions and 99% is a single
"interaction propensity." Different mode each time, but always the same structure:
one dominant, **feature-agnostic** component, and a thin feature-specific residual
underneath — about 1–2%.

So the map from geometry to function is real, but it is *almost flat*. Geometry
barely modulates function, because the model mostly **isn't listening to which
specific feature you perturbed.** Poke feature #4,000 or feature #18,000 and, to
first order, the same thing happens. The 1–2% that does depend on the feature is
where I kept fishing for signal — and where the artifacts lived.

I spent the whole project deflating that 99% away as a nuisance to get at the
residual. It was the other way around. The 99% *was* the finding: the model's
response to individual features is dominated by something that doesn't care about
their identity.

## Two reasons this was nearly invisible

**I was studying a lens, not the object.** An SAE is a re-description of the
model's *activations*. But function lives in the *weights* — the actual
computation. I was asking whether the geometry of an activation-decomposition
predicts perturbation responses, when the machinery that would connect them lives
one level down, in the mechanism the SAE doesn't capture. The months result is the
tell: the SAE's geometry just *mirrored* the model where the model had structure,
and *decoupled* where it didn't. It was never contributing a geometry of its own —
it was a shadow of the model's.

**Geometry and function are two shadows of the same object.** Both are derived from
the model. So any correlation between them has to flow *through* the model — and the
model's dominant behavior (that shared response) drowns the direct link. I was
hunting for a law connecting geometry to function when there was only a common
cause, mostly washed out. Shadows of a thing don't determine each other. Only the
thing does.

## Why this isn't a failure

The word "failed" smuggles in an assumption: that the treasure was there and I
missed it. But the honest read is that **I proved the map was mostly blank, and
showed why it's blank** — the model's response to a feature doesn't vary
feature-to-feature the way a rich functional geometry would require. That is a
real, non-obvious fact about how these networks work, and it's more useful than the
positive I wanted, because the positive would have been the field's comfortable
assumption confirmed, and this is the assumption falsified.

The only thing that actually failed was the premise I started with — the shared
belief that SAE feature geometry is a rich functional map. Killing that belief,
carefully, against the right nulls, *is* the result.

## The method, stated once

If there's one transferable thing across all three posts, it's this. I could never
*be right* — every attempt to prove geometry organizes function died. What worked
was the inverse: try to *be wrong less*. Prove what geometry *can't* do, as
rigorously as you can, and let the pile of eliminations pin down the answer. Every
dead positive ruled out a wrong hypothesis; what survived elimination — a
global per-feature magnitude does the organizing, geometry adds a thin real sliver
via interactions — is the answer, arrived at by subtraction, not discovery.

The load-bearing move was never a metric. It was pointing the null at my own good
news, first, harder than I pointed it at anyone else's. The negative shape was the
shape of the truth the whole time. Only inversion could see it.

## Coda

The map was mostly blank. But I can tell you, now, *why* it's blank — and in a
field where it is very easy to publish a beautiful number a random matrix would
have drawn for you, the blank map with a reason is the more honest artifact. I'll
take it.

---

## Correction (July 5, 2026)

Two days after publishing this, I tested the claim that carries this post — and it
broke. Reporting that here rather than editing the text above.

**What this post claimed:** the geometry map is blank *because* the model's response
to any feature is ~99% one feature-agnostic mode — "the model isn't listening to
which feature you touched."

**What the new measurements show** (GPT-2 vs Gemma-2-2b vs a same-family Pythia
ladder at 70m/160m/410m with one shared SAE recipe; code and numbers in the
[repo](https://github.com/OE-GOD/feature-geometry-locality)):

- The size of that shared mode tracks the **SAE training recipe**, not the network:
  ~69% for the ReLU SAE this post used, ~7% for JumpReLU on Gemma, ~5% for top-k on
  Pythia (magnitude-controlled, matched feature counts).
- In the controlled Pythia ladder it is **flat across a 6× scale range** — it is not
  a fact about model size either.
- On Gemma, the response is largely feature-*specific* (7% shared) — and geometry
  **still** fails to predict function there. So feature-agnosticism cannot be the
  reason the map is blank.

**What survives:** the negative itself, strengthened — geometry predicts function
below matched-null level at every scale, family, and SAE recipe tested. And the
post's broader lesson survives in a sharper form than I intended: I wrote that I had
been "studying a lens, not the object." The ~99% number was exactly that — a
property of the lens (the SAE recipe), which I mistook for a property of the object.

The follow-up post has the full story:
[Was the map blank because the model was small?](/2026/07/05/sae-geometry-at-scale/)
