---
layout: post
title: "Attribution patching breaks at depth: a layer-adaptive fix"
date: 2026-06-02
---

# Attribution patching breaks at depth: a layer-adaptive fix

*Or: why the standard cheap approximation for SAE circuit discovery is wrong where it matters most, and a one-line change that gives a 5× speedup over activation patching at near-perfect accuracy.*

*June 2026 · [Code + figures](https://github.com/OE-GOD/sae-pythia-160m) · [Companion paper](https://github.com/OE-GOD/sae-pythia-160m/blob/main/writeup/sae_pythia_layer6_paper.pdf)*

---

## The setup

SAE circuit discovery — figuring out which downstream attention heads consume a given SAE feature — is currently bottlenecked on activation patching (AP). For every (feature, head, position) triple you want to test, AP needs one forward pass through the model. At SAE scale (16k features × ~100 heads × many positions) this is prohibitive.

The standard fix is **attribution patching (AtP)**: one forward + one backward pass per firing position estimates the patching effect for *every* head simultaneously via a first-order Taylor expansion. The Marks et al. *Sparse Feature Circuits* paper (2024) uses an upgraded variant — **integrated gradients (IG)** — as the practical method, on the assumption that integrating gradients along the clean→ablated path is universally better than AtP's single-point gradient.

I tested both on the same 180 (feature, head) pairs in Pythia-160M layer 6, with activation patching as ground truth. The result surprised me enough that I'm writing this post.

**Caveat upfront:** this is 180 pairs on one small model, three TRUE driver features, five downstream layers. The qualitative pattern is robust but the specific threshold ("use AP at the deepest layer only") is model-specific. Treat this as a hypothesis worth testing on Gemma 2 and Llama, not as an established law.

---

## What I found

Overall agreement between AtP and AP is high — Pearson r = 0.947 across all 180 pairs, sign agreement 96%. If you stopped there, you'd say AtP works.

It doesn't. Stratified by downstream layer:

| Downstream layer | Pearson(AtP, AP) |
|---|---|
| 7  (L+1 from SAE) | 0.9994 |
| 8  | 0.9979 |
| 9  | 0.9927 |
| 10 | 0.9367 |
| 11 (L+5) | **0.7822** |

AtP is essentially exact at the layer immediately downstream of the SAE and degrades monotonically with depth. At layer 11 — five layers downstream of the SAE — individual mediator magnitudes can be off by 14×. One feature's mediator at L11H8 has AP effect −0.0561 and AtP estimate −0.0038.

This is the saturation/curvature regime: after the SAE feature's contribution has propagated through several layers of nonlinearity, the metric's response to ablating that feature is no longer well-approximated by the gradient at the un-ablated point.

**Why this matters for circuit discovery.** Suppression circuits — the Negative Name Mover Heads of the IOI paper, the inhibition heads of indirect object identification — live deep in the model. An AtP-only pipeline applied at deep layers will systematically miss them or get their magnitudes badly wrong. The cheap method fails exactly where the interesting structure is.

![Per-layer AtP vs AP comparison, showing degradation with depth](/assets/figures/fig6_atp_vs_ap_comparison.png)

---

## Does integrated gradients fix this?

This is the obvious next question, because IG is the method Marks et al. use in Sparse Feature Circuits. With N = 10 alpha steps along the path from clean to fully-ablated:

| Layer | AtP | **IG** | AP |
|---|---|---|---|
| 7  | 0.9994 | **0.8926** | 1.0 |
| 8  | 0.9979 | 0.9764 | 1.0 |
| 9  | 0.9927 | **0.7786** | 1.0 |
| 10 | 0.9367 | 0.9155 | 1.0 |
| 11 | 0.7822 | **0.9693** | 1.0 |

Two things jumped out:

**IG is *worse* than AtP at early layers.** At L7, IG drops to 0.89 while AtP stays at 0.999. At L9, IG drops to 0.78 — AtP is at 0.99. The standard "IG is more accurate because it integrates along the path" intuition fails: where AtP is already nearly exact, the partial-ablation states IG samples through are slightly out-of-distribution and add noise rather than reducing it.

**IG only wins where AtP fails.** At L11, IG climbs to 0.97 while AtP collapses to 0.78. The two methods have *complementary failure modes*: AtP fails at depth due to curvature, IG fails at early layers due to OOD intermediate states.

Overall, IG sits at Pearson 0.947 — the same as AtP — but costs 10× more (20 model passes per feature-position vs 2). It is strictly Pareto-dominated by AtP at the overall level.

This is a real problem for the "use IG everywhere" framing. The right question isn't "which approximation is best?" — it's "where is each approximation accurate?"

---

## The fix: layer-adaptive patching

Once you see the complementary failure modes, the method writes itself. Define a single threshold T:

$$\hat{e}_{L,h} = \begin{cases} \text{AP}_{L,h} & \text{if } L \ge T \\ \text{AtP}_{L,h} & \text{if } L < T \end{cases}$$

Use AtP where it's accurate (early layers), AP where it's not (deep layers). I swept T:

| T (AP applied for L ≥ T) | Pearson | Cost vs full AP |
|---|---|---|
| 7 (all AP) | 1.0000 | 100% |
| 9 | 0.9994 | 60% |
| 10 | 0.9985 | 40% |
| **11** | **0.9919** | **20%** |
| 12 (all AtP) | 0.9466 | 0% |

**Best operating point: T = 11.** Apply AP only at the deepest layer, AtP everywhere else. Result: Pearson 0.992 with full AP at 23% of full-AP compute — a **5× speedup at <1% accuracy loss**. Total cost is 2 (AtP) + 12 (AP at L11 for 12 heads) = 14 model passes per feature-position, vs 60 for full AP.

The full four-way Pareto comparison:

| Method | Cost (passes/feature-position) | Overall Pearson with AP |
|---|---|---|
| AP (ground truth) | 60 | 1.000 |
| **Adaptive (T = 11)** | **14** | **0.992** |
| IG (N = 10) | 20 | 0.947 |
| AtP | 2 | 0.947 |

Adaptive Pareto-dominates IG on both axes — cheaper *and* more accurate. It Pareto-dominates AtP on accuracy (0.99 vs 0.95) at modest extra cost (14 vs 2 passes). And it gets to within 0.008 of full AP at 1/4 the cost.

![Four-method Pareto frontier comparing AtP, IG, Adaptive, and AP](/assets/figures/fig8_all_methods_comparison.png)

---

## What I think this means

**For practitioners doing SAE circuit discovery:** if you're using AtP or IG to score (feature, head) pairs and only verifying with AP after filtering, you're probably missing the deep-layer suppression circuits that the field is most interested in characterizing. Spend AP compute on the deepest 10–20% of downstream layers; use the cheap method elsewhere. The exact threshold needs re-tuning per model (Pythia-160M has 12 layers, so "deepest 20%" is the last ~2 layers; Gemma 2 27B has 46 layers, so "deepest 20%" is ~9 layers — still a big compute saving).

**For the IG-vs-AtP debate:** the right framing isn't "which approximation is better" but "where does each approximation work." The Marks et al. (2024) result — that IG outperforms AtP for circuit discovery — likely depends on the layer mix in their experimental setup. Per-layer benchmarks would help the field figure out where IG's path-integration genuinely buys accuracy vs where it just adds OOD noise.

**For interpretability methodology more broadly:** approximation methods deserve to be benchmarked stratified by the structural axis along which their assumptions can fail. "Pearson 0.95 overall" can hide a 0.78 at the most important layer. This is the same lesson Heimersheim & Nanda (2024) emphasize for activation patching itself — different protocols can give wildly different effect sizes — but it applies to the approximations too.

---

## Limits I'm aware of

- **n = 180 pairs, one model.** Three TRUE driver features × twenty heads × five downstream layers in Pythia-160M. Whether the same depth-degradation pattern holds in Gemma 2 27B or Llama 70B is an empirical question I haven't answered.
- **Three features is small.** All three are "newline driver" features identified in earlier work (Findings 8–9 of the [companion paper](https://github.com/OE-GOD/sae-pythia-160m/blob/main/writeup/sae_pythia_layer6_paper.pdf)). The pattern may differ for features in other semantic clusters.
- **IG was only tested at N = 10 alpha steps.** Higher N may shift the comparison. But the asymmetric failure pattern (IG bad at early layers, AtP bad at deep) suggests N alone won't fix it — a *layer-adaptive IG* (IG at deep layers, AtP elsewhere) is the obvious next thing to test.
- **AP itself is not ground truth in the deepest sense.** It's the strongest interpretability method available here, but a behavioral metric (does the intervention change generations in the predicted direction?) would be a more rigorous benchmark for all four methods.
- **The "use AP at the deepest layer only" recipe is model-specific.** The principle (cheap method early, expensive at depth) should generalize; the exact threshold won't.

---

## What I might do next

A few directions, in rough order of how excited I am about them:

1. **Replicate on Gemma 2 2B**, where there's a published SAE (Gemma Scope) and the depth-to-degradation curve can be characterized properly across 26 layers instead of 5.
2. **Test layer-adaptive IG** — IG at the deepest layer, AtP elsewhere. If IG's deep-layer Pearson (0.97) generalizes, this could be a 2-method adaptive that's almost as accurate as 3-method (AP at the bottom).
3. **Stress-test on suppression circuits specifically.** The IOI paper's Negative Name Mover Heads live at L10H7 and L11H10 in GPT-2 small. If I could replicate IOI on Pythia-160M and apply each method's mediator-discovery to those known heads, I'd have a direct test of the "AtP misses suppressors" claim.
4. **Submit to a workshop.** This is the kind of methodology result that fits well in a mech-interp workshop track at NeurIPS or ICLR. The contribution is small but concrete and reproducible.

---

## What I learned from doing this

The pattern that produced this finding was generic enough that I want to name it: **when a cheap approximation matches an expensive ground-truth method "on average," check whether the match holds stratified by the structural axes along which the approximation's assumptions can fail.** First-order Taylor approximations fail when nonlinearity dominates. Depth is one obvious axis for nonlinearity to accumulate. The check is a one-line `groupby(layer).corr()`.

The IG result was the genuine surprise. I'd assumed — because Marks et al. assume — that integrating along the path is monotonically better than taking the gradient at a single point. The mechanism for *why it's worse at early layers* (intermediate alpha values are slightly OOD; the gradient at those OOD states is noisier than the gradient at the clean state, where AtP samples) is, in retrospect, obvious. But I didn't see it until the per-layer table came out.

The constructive contribution — layer-adaptive patching — was the kind of method-shaped result that came out of two negative results stacking together. AtP failing at depth is a negative result. IG failing at early layers is another negative result. Adaptive is just "do the obvious thing once you've named both failure modes." The hard part wasn't the method; it was running the benchmark precisely enough to *see* both failure modes.

I think this generalizes to a useful research heuristic: when two methods both fail differently, the adaptive combination is almost always worth trying before you reach for a more sophisticated single method.

---

*Reproducible code, full numerical results, and figures at [github.com/OE-GOD/sae-pythia-160m](https://github.com/OE-GOD/sae-pythia-160m) — specifically `code/32_attribution_patching.py` through `code/36_compare_all_methods.py`.*

*Feedback welcome: [irving46764@gmail.com](mailto:irving46764@gmail.com)*
