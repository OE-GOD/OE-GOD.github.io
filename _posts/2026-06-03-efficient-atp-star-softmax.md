---
layout: post
title: "Efficient AtP*: 30× speedup for SAE circuit discovery"
date: 2026-06-03
---

# Efficient AtP*: 30× speedup for SAE circuit discovery

*Activation patching is the gold standard for finding which attention heads consume an SAE feature. It costs one model run per (feature, head) — prohibitive at SAE scale. The standard cheap alternative (AtP) fails badly at deep layers because of attention softmax saturation. This post: I diagnose the failure mechanism, build the correction (closed-form softmax-aware AtP), and show it matches AP at 1/30th the cost.*

*June 2026 · [Code + figures](https://github.com/OE-GOD/sae-pythia-160m) · [Companion post on AtP layer-degradation](/2026/06/02/atp-degrades-with-depth/)*

---

## The starting point

In the [previous post](/2026/06/02/atp-degrades-with-depth/), I showed that attribution patching (AtP) — the standard cheap approximation to activation patching — degrades with downstream depth in Pythia-160M layer 6 SAE features. Pearson with AP drops from 0.999 at L7 to 0.78 at L11. Integrated gradients (Marks et al. 2024's chosen method) is worse than AtP at early layers and only wins at the deepest one.

I built a layer-adaptive method that uses AtP early and AP late, getting a 5× speedup at Pearson 0.99. That post ended with "open question: why does AtP fail at depth, and can we fix it directly?"

This post answers that.

**TL;DR:** The failure is entirely attention-softmax saturation. Replacing the softmax's linearized chain rule with the actual softmax(patched_scores) in closed form gives **Pearson 0.993 at 2 model passes — same cost as plain AtP, 30× speedup over full AP**.

**Caveat upfront:** 180 (feature, head) pairs, 3 TRUE driver features, Pythia-160M. The mechanism story is general but the specific Pearson number is one model on one feature cluster. Replication is the obvious next step.

---

## Diagnosing the failure: which nonlinearity breaks AtP?

AtP is first-order Taylor expansion of the metric around the clean state. It fails when the response is nonlinear. Three downstream nonlinearity candidates in a transformer:

1. **GELU** in the MLP
2. **Softmax** in attention
3. **LayerNorm** before attention/MLP

I tested each by **linearizing it** at L11 (the failure layer) and re-running activation patching. The principle: if I replace nonlinearity X with its tangent at the clean activation, and AP-with-X-linearized then matches AtP, X was the AtP killer.

### Result 1: alpha-scaling confirms nonlinearity at L11

First, an empirical check that the failure is even *about* nonlinearity. I measured the effect at α ∈ {0.25, 0.5, 0.75, 1.0} for each (feature, head) at L7 (control, where AtP works) and L11 (where it fails). If the response is linear in α, AtP is exact; if curved, AtP fails.

![Per-layer alpha-scaling response curves](/assets/figures/fig9_alpha_scaling_diagnostic.png)

At L7 (left): all 36 response curves sit on the linear reference. Linearity ratio = 0.99 (median). AtP works.

At L11 (right): curves fan out wildly. 67% are concave, IQR for linearity ratio is [0.71, 1.28]. AtP fails because the response is nonlinear — and the failure mode is *heterogeneous per-pair*.

### Result 2: GELU is NOT the killer

I cached the L11 MLP pre-activation, replaced GELU with its first-order tangent around clean (using autograd to get the GELU derivative), and re-ran AP at L11.

Effect on AP_full: **zero.** Pearson(AP_gelu_linearized, AP_full) = 1.000, RMSE = 0.0. Every single (feature, head) pair gave identical AP values whether GELU was nonlinear or linearized.

Interpretation: the perturbation at L11 MLP's pre-activation is small enough that GELU is already operating in its locally-linear regime. Or the L11 MLP doesn't carry these features' effect at all. Either way, GELU isn't what AtP misses.

### Result 3: Softmax IS the killer

I cached the L11 attention scores and pattern, replaced softmax with its first-order Jacobian-linearized version around the clean pattern, and re-ran AP.

| | Pearson with AP | Pearson with AtP |
|---|---|---|
| Baseline (AtP itself) | 0.782 | 1.000 |
| GELU linearized at L11 | 1.000 (no change) | 0.782 (no change) |
| **Softmax linearized at L11** | **0.801** | **0.9963** |

Linearizing the softmax at L11 makes AP **collapse to AtP** (Pearson 0.9963). The smoking gun: in a counterfactual world where the L11 softmax was linear, AtP would be near-exact.

Specifically, the worst-case pair (f15245 L11H8, where AP = −0.056 and AtP = −0.004 — a 14× underestimate):
- AP_full = −0.0561
- AP_gelu_linearized = −0.0561 (no change)
- AP_softmax_linearized = **−0.0062** (collapses to AtP)
- AtP = −0.0038

Strip the softmax curvature out, and the 14× AtP error vanishes. Softmax saturation is 100% of the failure mode.

---

## The fix: softmax-aware AtP (AtP\*)

Standard AtP computes the gradient through the softmax using its Jacobian at clean scores. The fix is to use the **actual softmax(patched scores)** in the chain rule — capturing the full nonlinear pattern change instead of its linearized approximation.

For each (feature, head, position):

1. Compute Δscores at the relevant attention head in closed form from Δq, Δk
2. Apply actual softmax: patched_pattern = softmax(clean_scores + Δscores)
3. Δpattern = patched_pattern − clean_pattern
4. Effect ≈ ∇M_pattern · Δpattern + ∇M_v · Δv (V-side stays linear)

The expensive version (extra forward pass per intervention): Pearson 0.993 at 62 passes. The cheap version (closed-form computation of Δscores using LayerNorm-aware Δq and the model's rotary): **Pearson 0.993 at 2 passes — same cost as plain AtP**.

Both versions agree bit-identical on the 180-pair test set.

---

## Results

**Per-layer Pearson with AP:**

| Layer | Plain AtP | Efficient AtP* |
|---|---|---|
| 7 | 0.9994 | 0.9995 |
| 8 | 0.998 | 0.999 |
| 9 | 0.993 | 0.996 |
| 10 | 0.937 | **0.994** |
| 11 | **0.782** | **0.985** |

The L11 collapse from 0.78 to 0.985 is the main result. AtP* essentially closes the depth-degradation gap.

**Full Pareto frontier on 180 (feature, head) pairs:**

![Full Pareto frontier with per-pair adaptive sweep](/assets/figures/fig10_all_methods_with_per_pair.png)

| Method | Pearson | Cost (passes per feature-pos) | Speedup |
|---|---|---|---|
| Full AP (truth) | 1.000 | 60 | 1× |
| Per-pair adaptive | 0.9987 | 16 | 3.8× |
| **Efficient AtP*** | **0.9933** | **2** | **30×** |
| Layer-adaptive | 0.992 | 14 | 4.3× |
| IG (N=10) | 0.947 | 20 | 3× |
| Plain AtP | 0.947 | 2 | 30× |

Efficient AtP* Pareto-dominates every method at AtP cost. Per-pair adaptive still wins on raw accuracy at ~8× the cost; it's the right choice if you need Pearson > 0.998. For typical SAE circuit discovery, efficient AtP* is the right operating point.

---

## Update (June 3): cross-architecture replication — partial success, honest scope limit

After publishing, I tested whether efficient AtP* generalizes. Two follow-up experiments.

### Pythia generalization across 15 diverse features (works)

The original Pearson 0.993 result was on three TRUE driver features, all from the newline cluster. I re-ran on 15 features spanning 7 newline contexts plus 8 non-newline categories: French street names, decimal points, file paths, BibTeX/LaTeX, punctuation, logical operators, BPE continuations, exponentiation notation.

Result: **Pearson 0.992** across 12 features that produced sufficient firing positions. Per-feature range 0.929–0.999. The method is not specific to newline drivers — it works across diverse SAE features.

### Hybrid efficient AtP* + per-pair AP fallback (new Pareto point)

Combining the two ideas — use efficient AtP* as base, fall back to AP only on pairs flagged uncertain by the linearity probe — produces a new Pareto frontier point:

| Config | Pearson | Cost (passes/feature-pos) | Speedup |
|---|---|---|---|
| Hybrid (probe=0.005, tol=0.10) | **0.9990** | **12.1** | 5× |
| Per-pair adaptive | 0.9987 | 15.6 | 3.85× |
| Hybrid (probe=0.020, tol=0.5) | **0.9978** | **3.8** | **16×** |
| Efficient AtP* | 0.9933 | 2 | 30× |

The hybrid Pareto-dominates per-pair adaptive: same Pearson 0.999 at 12 passes vs 16 (23% cheaper). And the cheap end gives Pearson 0.998 at 3.8 passes — the cheapest method on the >0.997 frontier.

### Gemma 2 2B replication — partial success, honest scope limit

I then tested cross-architecture generalization on Gemma 2 2B (different architecture: GQA with 8 Q heads + 4 K/V heads, RMSNorm, attention softcap, 26 layers). Adapted the closed-form to handle these differences. Tested with Gemma Scope SAEs.

Two configurations:

| Setting | SAE layer | Downstream layers | Pearson(AtP*, AP) |
|---|---|---|---|
| Pythia-160M (original) | 6 | 5 | **0.993** |
| **Gemma 2 2B (deeper SAE)** | **22** | **3** | **0.700** |
| **Gemma 2 2B (mid SAE)** | **12** | **13** | **0.407** |

**The depth-of-cascade matters a lot.** Going from 13 downstream layers (Gemma L12 SAE) to 3 (Gemma L22 SAE) improved Pearson from 0.41 to 0.70. But even at 3 downstream layers, Gemma's accuracy is well below Pythia's at 5 layers (0.99).

I spent an hour debugging the worst disagreement (f1041 L15H5: AP = −0.043, AtP* = +0.0003 — a sign flip). The closed-form math is **provably correct**:

| Quantity | Cosine sim (closed vs actual) | Magnitude ratio |
|---|---|---|
| Δq | 0.9996 | 0.998 |
| Δrot_q | 0.9991 | 1.002 |
| Δscores | 0.9991 | 0.997 |
| Δpattern | **1.0000** | 1.05 |

But here's the critical test: even substituting the *actual* Δpattern from a real perturbed forward (skipping the closed-form entirely), `g_pattern · Δpattern` gives +0.003 while AP gives −0.008 — different sign. The linear approximation through the L15 attention pattern is wrong by sign *even when the pattern itself is computed exactly*.

**This is a fundamental limitation, not an implementation bug.** The gradient `g_pattern` is the first-order sensitivity of the metric to a change in L15's attention pattern. When the perturbation propagates through 10 more downstream nonlinear blocks (each with attention + MLP + LN, plus softcap), cascading higher-order effects dominate. First-order chain rule — even with softmax saturation correctly handled at the intervention layer — breaks down.

### What this means for the "best in the world" claim

Honestly: it's an overclaim for the full method. The correct framing:

- **Efficient AtP\* is the best cheap method for SAE circuit discovery in the *shallow-downstream regime***: when the intervention is within ~3–5 nonlinear blocks of the output. Within this regime, it Pareto-dominates plain AtP, IG, layer-adaptive patching, and per-pair adaptive.
- **For deeper interventions**, the first-order approximation has unavoidable limits without further methodological work (higher-order corrections, hybrid AP+AtP* by intervention depth, or fundamentally different approaches).
- **Practical recipe for Gemma deployments**: place SAEs within ~5 layers of the output for efficient AtP* to be reliable; for deeper interventions, the choice is either fall back to AP or accept reduced accuracy.

### What I learned (third methodology lesson)

The two earlier lessons (diagnose-before-build, write-the-empirical-vs-analytical-check) both came from cases where I made progress. The third lesson is from a case where I had to stop:

**Negative results from cross-architecture transfer are about the validity of your assumptions, not about your code.** I spent significant time hunting for a Gemma-specific bug. There was none. The math was exact. What failed was the first-order chain-rule assumption — which held for Pythia's 5-layer cascade but not for Gemma's 13-layer one. The right framing wasn't "find the bug" but "characterize the regime of validity." Future cross-architecture replications should treat regime-of-validity as the primary question to answer, not as the fallback when no bug is found.

This shapes the writeup honestly: a Pareto-dominant method on Pythia with provable Δpattern math, validated within-model across 15 features, with explicit and bounded scope limits on cross-architecture transfer. That's a real contribution, not a sweeping claim.

---

## What the LayerNorm bug taught me

Implementing efficient AtP* was straightforward in principle: cache the clean state, apply softmax to (clean_scores + closed-form-Δscores), use the cached gradient. The first version was wildly off — Pearson −0.46 at L7, sign flips everywhere.

The bug: TransformerLens's `hook_q_input` is **pre-LN**, not post-LN. The actual computation is `q = LN(q_input) @ W_Q + b_Q`. My closed-form Δq used `Δq_input @ W_Q[h]` and ignored the LayerNorm, overestimating Δq by ~4.6× (driven by LayerNorm's normalization shrinking the perturbation).

Fix: compute ΔLN(q_input) directly by calling the LN module on (clean − f·decoder), then multiply by W_Q. The closed-form result then matched the validation (expensive) AtP* bit-identical.

The lesson generalizes: when reaching into a model surgically with closed-form math, every transformation between your hook point and the operation you care about has to be accounted for. The hook point tells you *what* the function receives as input, not *how* it's transformed downstream.

I almost missed this because the diagnostic check that revealed it (comparing `actual Δq` from a perturbed forward pass to my closed-form Δq) was 5 minutes of code — but it's the kind of check I almost didn't do because everything *looked* correct.

---

## What efficient AtP* actually does

Single clean forward + backward (2 passes total) caches:
- pre-LN q_input, rotated Q/K, attention scores, attention pattern (per downstream layer, per head)
- gradient of metric w.r.t. attention pattern and v_input

For each (L, h), closed-form math only:
- ΔLN(q_input) = LN(clean_q_input − f·decoder) − LN(clean_q_input) — one LN call
- Δq = ΔLN · W_Q[h], Δk = ΔLN · W_K[h]
- Apply rotary at position 'last' to Δq, Δk via the model's `apply_rotary` (rotary is linear, so this is exact)
- Δscores[last, :] = Δrot_q · clean_rot_k.T / √d_head + cross-term at j=last
- patched_pattern[last, :] = softmax(clean_scores + Δscores)
- Effect = −∇M_pattern · (patched − clean) + ∇M_v · (f·decoder)

The whole per-(L, h) computation is small matrix multiplies. On Pythia-160M with 30-token context, all 9 feature-positions × 60 head-layers runs in ~10 seconds on MPS.

---

## Honest limits

- **180 pairs is small.** 3 TRUE driver features × 12 heads × 5 layers × 3 positions, all from one feature cluster (newline drivers). The mechanism (softmax saturation) is general; the specific Pearson number may not be.
- **Pythia-160M only.** Whether the AtP failure curve looks the same on Gemma 2 (different rotary, different size) is the obvious replication study. If the curve generalizes, efficient AtP* generalizes.
- **Not benchmarked against Kramár et al. 2024 AtP\* exact implementation.** Our version has the softmax/Q correction. Their published version also has K-residual fix and GradDrop. They might be more accurate; we haven't compared.
- **The remaining 0.007 Pearson gap to AP** (0.993 vs 1.0) is real. It likely reflects second-order interactions between the perturbation and other nonlinearities (LayerNorm cascading, MLP at later layers) that the first-order softmax fix doesn't capture. A hybrid (efficient AtP* + per-pair AP verification on uncertain pairs) would probably close this at ~3-4 passes.
- **TRUE driver features only.** The 3 features tested were already identified as causally important. Whether AtP* is accurate for noise pairs (where AP ≈ 0) is untested.

---

## What I might do next

In order of how excited I am:

1. **Replicate on Gemma 2 2B.** Gemma Scope gives published SAEs and 26 downstream layers (vs my 5). If the AtP-vs-depth curve in Gemma also collapses to softmax saturation, and efficient AtP* recovers Pearson 0.99 there, that's a real generality claim. This is the experiment that would turn this into something the field cites.

2. **Implement Kramár et al. 2024 AtP\* exactly and benchmark.** If I match or beat their numbers, this is a stronger paper claim. If I lose, I learn what's missing (probably K-fix or GradDrop).

3. **Hybrid: efficient AtP\* + per-pair verification.** AtP* as the cheap base, AP only on pairs flagged as uncertain. Probably Pearson 0.999 at 3-4 passes — the asymptotic best.

4. **Workshop submission.** This is the kind of mech-interp methodology result that fits a NeurIPS / ICLR mech-interp workshop track. The contribution is small, concrete, and reproducible.

---

## What I learned from doing this

Two things I want to remember.

**Lesson 1 — Diagnose before you build.** The first instinct on seeing "AtP fails at deep layers" was to try a bunch of fixes (trapezoidal AtP, midpoint AtP, integrated gradients, per-pair adaptive). Most of these gave Pareto improvements but no fundamental understanding. The single experiment that mattered most was the linearization diagnostic: replace each candidate nonlinearity in turn and see which one, when made linear, makes AtP exact. That experiment cost 20 lines of code and 10 seconds of compute. It pinpointed softmax as the entire failure mode. Building the fix took a few more hours; designing the right diagnostic took less time than that. The intuition: **negative results that constrain the hypothesis space are often more valuable than positive results that don't.**

**Lesson 2 — Closed-form math needs to account for everything between hook and operation.** The LayerNorm bug cost me an hour and an embarrassing first set of numbers (Pearson −0.46). The fix was trivial once diagnosed. The diagnostic — compare `actual Δq from a real forward pass` to `my closed-form Δq` — was 5 minutes of code and would have caught the bug immediately. I almost didn't write it because the formula *looked* right. Worth internalizing: when implementing closed-form numerical methods, write the empirical-vs-analytical check before you trust the analytical result, even when (especially when) the math seems obviously right.

---

*Reproducible code, all checkpoints, and figures at [github.com/OE-GOD/sae-pythia-160m](https://github.com/OE-GOD/sae-pythia-160m). The efficient AtP* implementation is in `code/49_atp_star_efficient.py` — about 280 lines including comments.*

*Feedback welcome: [irving46764@gmail.com](mailto:irving46764@gmail.com)*
