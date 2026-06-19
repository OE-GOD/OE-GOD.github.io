---
layout: post
title: "Are SAE features real? Stop certifying features, validate detectors"
date: 2026-06-17
---

# Are SAE features real? Stop certifying features, validate detectors

*I tried to answer "are sparse-autoencoder features real?" and found the question, as usually asked, is unanswerable — there's no oracle for which features are real, and when you test it, most individual features fail. But the useful version is answerable: you can build, validate, and deploy feature-based **detectors**, certified by held-out and cross-distribution task performance. This post walks the whole arc — from the no-oracle wall to a deployed detector that scores perfect precision on unseen prose and fails safe on code.*

*June 2026 · [Code + figures](https://github.com/OE-GOD) · built from scratch by hand, every number from a reproducible script*

---

## Why I rebuilt everything from scratch

I had a portfolio of interpretability work that AI had largely typed for me. I could *recognize* it but not *reproduce* it — and that gap showed the moment I had to explain it under pressure. So I rebuilt the core by hand: trained SAEs on a language model, typed every line, justified every step before running it. This post is what that rebuild found when I pointed it at one question: **are the features real?**

---

## Finding 1 — Most individual features don't replicate

Train two SAEs on the same model activations, identical except the random seed. If a feature is a real property of the model, both runs should find it.

- Toy SAE (Pythia-160M layer 6, 2048 features): both SAEs reached **identical loss (0.1056)**, yet only **1.0%** of features had a cross-seed twin at cosine > 0.9.
- My earlier 16k study found **2.14%** across seed, width, and architecture.

![Each feature's best-match cosine to a second SAE; the mountain sits far left of the 0.9 threshold](/assets/figures/realness-replication.png)

Equally good at the job; almost no shared vocabulary. **Most "features" are artifacts of one training run.**

But this does *not* mean the model has no real concepts. The information is there — it's **smeared across many features and sliced differently each seed**. The concept is real; the single-feature label for it usually isn't.

---

## Finding 2 — "Real" is not one property. It's several, and they disagree.

I operationalized five independent tests of "real":

| Axis | Question | How |
|---|---|---|
| Stability | does it replicate across seeds? | cross-seed decoder cosine |
| Necessity | does removing it change output? | ablation (per-firing) |
| Sufficiency | does adding it induce the concept? | steering |
| Interpretability | does it fire on one consistent thing? | top-token concentration |
| Logit-coherence | clean direct effect on output tokens? | decoder · unembedding |

Then I asked whether the axes agree. They mostly don't.

- **stability vs causation:** weakly negative (about −0.2 at high N; stronger on Gemma). The features that replicate are often the ones the model uses *least*.
- **interpretability vs stability:** about 0 (independent). A feature can fire 100% on newlines (perfectly interpretable) yet be totally unstable (cosine 0.17).
- **logit-coherence:** stands alone, independent of interpretability and causation.
- **necessity ≠ sufficiency:** ablating and steering measure different things.
- **decoder-norm geometry:** stability +0.49, frequency −0.48 — big-norm directions are stable but causally inert; small-norm are the causal ones.

A feature can pass one test and fail another. **One test is never enough.**

(A noise warning I had to heed: at n=40, individual correlations wobbled ±0.3 — interpretability-vs-causation went +0.37 → +0.06 between runs. I re-ran the cheap axes at n=723 for trustworthy numbers, and walked back my own earlier "strong" claims to "weak.")

---

## Finding 3 — Cheap signals predict *stability* but not *causation*

Firing frequency (free, one SAE) predicts stability moderately — Pearson **0.52** on Pythia, **0.54** at 5 seeds, **0.37** on Gemma — via an inverted-U: rare features are hyper-specific fragments, common ones are incoherent background, clean concepts sit in between.

![Stability by firing-frequency quartile: rises sharply, peaks mid-high, dips at the top](/assets/figures/realness-invertedU.png)

But frequency does **not** predict causation (confound-controlled per-firing: about −0.1). **You cannot cheaply tell whether the model uses a feature.** The causal axes are independent of every cheap signal — they must be measured directly.

---

## Finding 4 — The no-oracle wall

To certify a test, you validate it against ground truth. For "real features," there is no ground truth — that's the question itself.

- On **synthetic** data where I planted known features, the battery worked: stability test **precision 1.00**, recall 0.62; necessity precision 0.70, recall 0.82.
- On **real** activations with injected ground truth, precision **collapsed to 0.04** — not because the test failed, but because real activations contain the model's own *unlabeled* real features, which the test correctly flags but my oracle can't label. And only **6/15** injected-real directions were even recovered amid superposition.

This is the wall, demonstrated: **you can't measure certifier precision on real features, because validating it requires the complete list of real features you're trying to find.** A full four-axis battery certified **0/12** toy features.

---

## The pivot — from feature-realness to detector-validation

If you can't certify individual features, change the question. Don't ask "is this feature real?" Ask "does this **validated combination** of features reliably do a task?" A task has labels — the labels are the oracle. This sidesteps the wall entirely.

**The recipe:** define a labeled task → get SAE activations → split held-out → select top-k correlated features → fit a sparse probe → tune threshold on train → judge on held-out → certify only if it clears the bar.

- Single features failed (digit recall 0.14 — concept splitting).
- Sparse probes (top-30 features + logistic regression) on the **real, deployable** Gemma Scope SAE certified **3/5** detectors at held-out F1 > 0.8: digit 0.83, newline 0.88, space 0.85.

---

## Industrial validation — detectors must hold across distributions

Held-out same-distribution isn't enough for deployment. I tested Pile-built detectors on TinyStories (a different distribution).

| Detector | in-dist F1 | OOD F1 | Verdict |
|---|---|---|---|
| newline | 0.88 | 0.98 | CERTIFIED (holds cross-distribution) |
| space | 0.85 | 0.98 | CERTIFIED |
| digit | 0.84 | 0.00 | **REJECTED** — no digits in OOD; would break in deployment |

The cross-distribution test **caught a detector that passed in-distribution but would have silently failed in production.** Catching that is the entire value of validation.

---

## Real-use deployment — works in scope, fails safe out of scope

I loaded the saved detector and ran it on fresh input.

- Fresh unseen Pile prose: **precision 1.00, recall 1.00.**
- Python code (untested domain): **precision 1.00, recall 0.40** — degrades, but **fails safe**: no false positives, just misses code newlines (structurally different).

Refined insight: distribution shift hurts in proportion to **token-structure** difference, not topic. prose → prose (TinyStories) transfers (0.98); prose → code does not (0.57).

---

## Where this fits — and what's new vs SAEBench

Most of this overlaps with work the field has already done, and I want to be precise about credit. The reference evaluation suite is **[SAEBench](https://arxiv.org/abs/2503.09532)** (Karvonen et al., ICML 2025) — 8 metrics across reconstruction, concept detection (sparse probing, absorption), interpretability (auto-interp), and disentanglement (RAVEL, SCR, TPP, unlearning). My detector-validation is essentially its **sparse-probing** eval; my "hate detector learned topic, not hate" is its **SCR** (spurious-correlation) setting; my replication finding is **[Paulo & Belrose (2025)](https://arxiv.org/abs/2501.16615)**, who showed only ~30% of features survive a seed change on Llama. I reinvented a lot of this independently — which I take as a sign the instincts are right, not as novelty.

So what's actually new here is narrow and specific. SAEBench evaluates SAEs **in-distribution and single-seed**. Two things it does not test:

1. **Cross-distribution validation.** A detector can pass in-distribution sparse-probing yet collapse out-of-distribution — `digit` went 0.84 → 0.00, `upper` went 0.84 → 0.24 on a new corpus. In-distribution success can be a spurious shortcut; **cross-distribution generalization is what separates real feature signal from a dataset-specific trick.** This is a necessary check the benchmark omits.

2. **The no-oracle framing.** Because there is no ground-truth list of "real features," you cannot certify individual features with precision guarantees on real models (I show this empirically: injected ground truth makes precision unmeasurable). The honest response is to validate *detectors* against a task — where the labels are the oracle — rather than features against nothing.

Honest calibration: the cross-distribution effect is **concept-dependent** — rare for surface concepts (1 of 7 token-level detectors failed cross-distribution), more common for semantic ones. So the claim is *"cross-distribution validation catches spurious detectors that in-distribution evaluation misses, especially for semantic concepts"* — not "in-distribution evals are worthless." It's a necessary addition to SAEBench's axes, not a replacement.

---

## Conclusion

> **Individual SAE features mostly aren't real — but the model's concepts are. They live in combinations of features, not single units. So don't certify features; build, validate, and deploy feature-based detectors — judged by held-out and cross-distribution task performance, documented with their scope and failure modes.**

This turns an unanswerable ontological question ("are features real?") into an answerable engineering one ("does this validated detector work, and where?"). The shift from **feature-realness to detector-validation** is the contribution.

## Honest limitations

- Toy/simple concepts (newline, digit, space). High-stakes concepts (toxicity, PII) are harder and need their own labeled tasks — but the *method* is identical.
- One SAE, one layer; cross-distribution tested on two OOD corpora (TinyStories, wikitext).
- This certifies a *detector's* reliability, not that its constituent features are individually "real" — a deliberately weaker, honest claim.
- The no-oracle wall is fundamental, not an engineering gap: native-feature realness cannot be certified with precision guarantees on real models.
