---
layout: default
title: Aung Maw
---

# Aung Maw

Solo researcher working on **mechanistic interpretability of language models** — currently focused on causal evaluation of sparse autoencoder (SAE) features. Background in software engineering; community-college path. Cracked.

Currently looking for: MATS, Apollo Research, FAR.AI, or equivalent mech-interp collaborations.

---

## Selected Work

### [Auto-interp Labels Conflate Driver and Thermometer SAE Features](/2026/06/01/sae-pythia-160m/)
*June 2026* — A seventeen-experiment characterization of TopK SAE features in Pythia-160M layer 6. The driver/thermometer split between SAE features auto-interp labels as "monosemantic" is a categorical property robust across intervention method and magnitude. Even among true drivers with identical labels, per-head path patching reveals features route through different downstream attention heads.

Three headline findings:
1. **Most "monosemantic" features are thermometers, not drivers.** 14 of 23 features showed zero drift toward their labeled concept when steered.
2. **Single-direction patching over-counts drivers.** Combining noising (necessity) with denoising (sufficiency) drops the true driver rate from 56% to 33%.
3. **Same-labeled drivers route through different heads.** Per-head path patching + causal validation via steering+ablation: mediator effects 30–44× larger than random-head ablation.

Solo work, $50 compute on a MacBook, full reproducible pipeline.

→ [Blog post](/2026/06/01/sae-pythia-160m/) · [PDF paper](https://github.com/OE-GOD/sae-pythia-160m/blob/main/writeup/sae_pythia_layer6_paper.pdf) · [GitHub repo](https://github.com/OE-GOD/sae-pythia-160m)

---

## Links

- **GitHub:** [github.com/OE-GOD](https://github.com/OE-GOD)
- **Email:** irving46764@gmail.com
- **LinkedIn:** *(post your profile URL here)*

---

## What I'm thinking about

- How to scale per-head path patching of SAE features from Pythia-160M to Gemma 2 / Llama 3
- Whether the 2×2 necessity × sufficiency framework generalizes to crosscoder SAEs (multi-layer features)
- Why same-labeled SAE features route through different downstream attention heads — is this geometry, training-data heterogeneity, or something else
- Property-based testing for ML systems (current side project — SRSE 2026 with Marcelo d'Amorim)

Feedback, collaboration, and cold emails all welcome.
