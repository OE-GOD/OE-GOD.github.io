---
layout: post
title: "SAE geometry, part two: measure the interactions, not the features"
date: 2026-07-02
---

# SAE geometry, part two: measure the interactions

*A follow-up to [the negative result](/2026/07/02/sae-geometry-does-not-predict-function/).
That post showed, five ways, that flat SAE decoder geometry doesn't predict
function — every "local halo" was a spectrum artifact a random matrix reproduced.
This post asks the question that survives: if a feature's function is how it
**combines** with other features (not what it does alone), is that combination
structure local or global? The answer is a real, network-specific local signal —
the first thing in the whole project to survive the random-matrix control — that
is nonetheless too thin to rescue local interpretability.*

*July 2026 · [Code](https://github.com/OE-GOD/feature-geometry-locality) ·
GPT-2-small + jbloom residual SAEs · adversarially verified, 3 skeptics + judge*

---

## The mistake the first post was still making

The first post killed five function proxies — direct logit effect, causal
injection, nonlinear ablation, co-firing. Every one measured a feature's effect
**in isolation**: ablate it, or inject it, and watch what changes. And every one
was ~99% dominated by a single shared mode (ablate *any* feature and the model's
distribution collapses toward glitch tokens — a generic damage response, not the
feature's specific role).

Run the Five Whys on that and you hit a root cause the whole project had walked
past: *the network never uses features in isolation.* The nonlinearities — the
MLP, attention — are exactly the machinery that makes features **interact**. A
feature's function is not what it does alone; it's who it works with. Measuring it
solo throws away the actual object.

Here's the soccer version. You want to understand a midfielder. You could ask
"what happens to the team if they sit out?" — but benching *anyone* hurts (the
shared mode), and more importantly a player's role isn't their solo stats, it's
**who they pass to**. So stop benching players. Watch the passes.

## Measuring the passes

The clean way to measure whether two features interact is the MLP's **second-order
coupling** — how much the MLP's response to both features present exceeds the sum
of each alone, on top of real residuals:

```
interaction(i, j) = ‖ mlp(h₀ + dᵢ + dⱼ) − mlp(h₀ + dᵢ) − mlp(h₀ + dⱼ) + mlp(h₀) ‖
```

This is zero if the MLP treats the features independently, nonzero where it
genuinely mixes them. It lives entirely in the gelu nonlinearity, so — unlike a
linear read — it isn't forced to track geometry by construction.

Then the local-vs-global question, made precise: **do geometrically-close features
interact more (local) or is interaction geometry-independent (global)?**

## The result, and the control that matters

Geometrically-close features *do* interact more: correlation with cosine geometry
+0.14, and the interaction matrix is — of course — 99% one shared "interaction
propensity" mode, so once you remove that per-feature propensity the pairwise
signal rises. But a positive correlation meant nothing five times in the last
post, so the only number I actually trust is the control:

**Random directions pushed through the same MLP give zero.** Spectrum-matched
random "features" — the exact control that reproduced every flat-geometry halo and
refuted it — show correlation ≈ 0 (−0.01, not significant) in all twelve
layer×seed configurations and under every control formulation. The real features
show a genuine effect; the random ones show nothing. **This is the first result in
the whole project to survive that control.** It's network-specific: close *real*
features interact more, and it isn't a generic "aligned directions share curvature"
artifact.

It's also not near-duplicate features (feature-splits interacting with themselves):
there are ~0 near-duplicate pairs, and excluding everything above cosine 0.3 barely
moves it. The signal lives in genuinely distinct features.

## But I over-reported it, and the skeptics caught me

I first wrote the effect size as +0.30 (the correlation after removing per-feature
propensity). An adversarial pass — three skeptics each running their own
experiments — showed that +0.30 is a Pearson number inflated roughly 2× by a few
high-leverage outlier pairs. The distribution-free (rank) estimate is **+0.15 to
+0.17**. The honest local effect is about half what I first claimed. Reporting the
+0.30 would have been a smaller version of the exact mistake the first post is
about.

And the deeper narrowing: the local component is real but **operationally
negligible**. Geometry explains ~2% of interaction variance; a single **global**
per-feature "how much does this feature interact with everything" scalar explains
~79%. In a direct test — recover which features a given feature actually interacts
with most — geometric neighbors reach precision 0.24 (better than chance 0.10), but
the global scalar reaches 0.54, winning about 2-to-1 in eleven of twelve configs.

Back to soccer: players *do* pass to nearby teammates a little more than chance —
but if you want to predict who someone passes to, one global "how involved is this
player" number beats "who's standing near them" two to one.

## The answer

For GPT-2-small, SAE feature geometry reflects the network's functional structure
**only weakly, and mostly globally** — a small, real, network-specific local
component that shows up in *interactions* (not in flat geometry), but is dominated
by a global per-feature propensity. **Global-sufficient, local-insufficient.** You
cannot infer who a feature works with from its geometric neighbors nearly as well
as from one global number.

That is not a solved open problem, and it is not a foundation for
bag-of-features-style *local* interpretability. But it is a real, controlled,
quantified answer to a question that gave only artifacts every other way I asked
it — and the load-bearing move was the boring one: measure the relation the
network actually computes with, then try as hard to break the positive as I'd
tried to break the negatives. The +0.30 that became +0.16 is that discipline
working. It's the part worth keeping.

## Coda

Two posts, one lesson, stated twice: the result was never the metric. It was the
null, and the willingness to point it at your own good news.
