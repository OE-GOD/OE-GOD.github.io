---
layout: post
title: "Was the map blank because the model was small? A scale ladder, and a claim I killed"
date: 2026-07-05
---

# Was the map blank because the model was small?

*A follow-up to a three-post series on SAE feature geometry.
[Part one](/2026/07/02/sae-geometry-does-not-predict-function/) killed five ways of
showing decoder geometry reflects function. [Part two](/2026/07/02/sae-geometry-part-two-interactions/)
found the one thin signal that survived. [Part three](/2026/07/03/why-the-map-was-blank/)
tried to explain why the map was blank — and this post corrects it (there is now a
dated correction block on that post). Here I take the sharpest objection to the whole
series — scale — and report both what survived it and a shiny new claim I made,
believed for about an hour, and then killed with the right experiment.*

*July 2026 · [Code](https://github.com/OE-GOD/feature-geometry-locality) — headline
numbers are from commit `50176c2` (`gemma_scale.py`, `gemma_cofire.py`,
`pythia_ladder.py`)*

---

## The doubt worth taking seriously

Everything in the series ran on GPT-2-small: 117 million parameters. The strongest
objection to the whole project is a threshold argument: **maybe rich functional
geometry only appears past a certain scale, and 117M is below it.** Induction heads,
in-context learning, grokking — the field's landmark phenomena are all threshold
effects, absent below some size and present, qualitatively, above it. If SAE feature
geometry is one of those, my blank map is just a photograph of a sub-critical pile.

So I tested it — and I want to show you the test *including* the part where I got it
wrong, because that's the honest shape of what happened.

## Step one: a 17× jump, and a result I loved

I ported the battery to **Gemma-2-2b** — ~17× larger — with Gemma Scope SAEs, matched
feature counts (K=400), mid-depth layers, and the same matched-null discipline as the
rest of the series.

First, the news that held: decoder geometry *still* doesn't predict genuine function.
Measuring co-firing (which features actually activate together on real text),
geometry predicts it *worse* than a matched random baseline at 2B, just as at 117M.
The decoder even got *more* isotropic with scale (top-1 singular energy 0.025 →
0.008) — thinner geometry, not richer.

Second, the shiny thing: the "shared mode" I'd made the centerpiece of part three —
the ~99% feature-agnostic response, "the model isn't listening to which feature you
touched" — **collapsed from 69% to 7%** going from GPT-2 to Gemma (on a
magnitude-controlled, binary co-firing measure). That looked like a discovery. It let
me write a clean story: at scale the model becomes feature-specific, so part three's
mechanism was a small-model regime. I drafted exactly that.

Then I remembered my own rule.

## Step two: the rule, applied to my own good news

The whole series has one method: point the null at your own result first, harder than
at anyone else's. And this "shrinks at scale" claim had a gaping hole — GPT-2 and
Gemma differ in *everything*, not just size. Architecture, vocabulary, tokenizer,
training data, and the SAE recipe (GPT-2's is ReLU; Gemma Scope's is JumpReLU).
"Scale" was confounded with five other things. I had no business calling it scale.

The fix is a **same-family ladder**: hold architecture and SAE recipe fixed and move
*only* size. So I ran Pythia — 70m, 160m, 410m — with EleutherAI's SAEs, one shared
top-k recipe at every rung. Now scale is the only variable.

Here's what the shared mode did across a 6× scale range (measured, binary,
magnitude-controlled, matched K=400):

| Pythia | shared mode |
|---|---|
| 70m | 0.047 |
| 160m | 0.037 |
| 410m | 0.078 |

**Flat.** It does not shrink with scale. It doesn't do anything with scale.

The 69% → 7% "collapse" wasn't scale at all — it was the SAE recipe. ReLU SAEs carry
a big shared activity axis (~69%); JumpReLU carries a small one (~7%); top-k a tiny
one (~5%). Same *kind* of number, three *recipes*, and it tracks the recipe, not the
network and not the size. My hour-old claim was measuring my instrument and calling
it a finding. The ladder killed it.

## What actually survived

Strip away the claim I killed and one thing is left standing — taller than before.
Decoder geometry does not predict co-firing, and now I can say it across three model
families, three SAE recipes, and five scales (measured; z is the distance from a
matched-null distribution, negative means *below* the nulls):

| model | geometry → co-firing (vs matched null) |
|---|---|
| GPT-2 (117M) | z = −16 |
| Gemma-2 (2B) | z = −7 |
| Pythia-70m | z = −55 |
| Pythia-160m | z = −37 |
| Pythia-410m | z = −40 |

Negative everywhere. The threshold hypothesis — that geometry becomes a functional
map once the model is big enough — found no support anywhere I could measure.
Seventeen times the parameters, a different architecture, three different SAE
recipes: the map stays blank.

## The part that stings, stated plainly

Part three told a mechanism story: the map is blank *because* the model's response is
~99% feature-agnostic. I now think that mechanism was wrong twice over. The ~99% was
an artifact of one SAE recipe (ReLU), not a fact about the network — a different
recipe on the same kind of measurement gives 7%, not 99%. And even where the response
*is* feature-specific (Gemma, 7% shared), geometry still fails to organize it. So
feature-agnosticism was neither universal nor the cause. It was a shadow the
instrument cast.

The uncomfortable summary: every time I reached past the negative to explain *why* —
"feature-agnostic response," "shrinks at scale" — I was describing the SAE, not the
model. The only claim that has survived every null, every recipe, every scale, and
every family is the negative itself. I keep trying to say something more interesting
than "geometry is not a map of function," and the experiments keep handing me back
the same flat sentence.

## Caveats

- The ladder tops out at 410m (and 2B cross-family). Nothing here rules out a
  threshold *above* 2B; "scale-invariant" means invariant over the tested range.
- The 410m rung uses MLP-site SAEs (EleutherAI publishes no residual SAE at that
  size); 70m/160m replicate the result at both residual and MLP sites.
- Function is operationalized as co-firing. Other notions of function (causal
  effects, interactions) were tested in earlier posts on GPT-2 only.
- One mid-depth layer per model, K=400 features per rung.
- What would overturn this: any rung, any recipe, where decoder geometry predicts a
  non-tautological function measure *above* a spectrum- or rotation-matched null.

## Coda

I set out to check whether my blank map was an artifact of a small model. It isn't —
the map is blank at 2B and across a clean scale ladder, more convincingly than
before. Along the way I found a beautiful "it changes at scale" result, believed it
for an hour, and dismantled it with the same move the whole series is built on. The
negative got stronger; two of my explanations for it died, and one of them was
published, so part three now carries a dated correction rather than a quiet edit.

In a project about pointing the null at your own good news first, the sharpest
correction was the one that cost me the shiniest claim in the draft. I'll take the
trade.
