---
layout: post
title: "Killing all five famous induction heads leaves most of GPT-2's induction standing"
date: 2026-07-29
---

# Killing all five famous induction heads leaves most of GPT-2's induction standing

*GPT-2 small's induction ability is not the five heads the literature names. It is spread
across independent spares, at least one backup that steps up when the five die, and
downstream consumers that partially die with them.*

## TL;DR

- I replicated the five famous induction heads in GPT-2 small — L5H1, L5H5, L6H9, L7H2,
  L7H10 — as the top five induction scorers (0.76–0.88). Nothing new there; that was the
  calibration step.
- Zero-ablating the strongest one raises repeated-sequence loss by +0.45. Zero-ablating a
  control head with no induction score raises it by +0.20. Half of a naive "ablation
  effect" is generic damage, not induction-specific.
- Killing all five together takes loss from 1.10 to 3.86 — worse than the sum of single
  ablations (~2.25), so the five partially cover for each other. But the no-repetition
  ceiling is 12.36. Most of the model's induction ability survives the famous five.
- The survivors sort into three species: an **independent** inductor that doesn't react at
  all (L5H0, score 0.49 → 0.49, identical in every seed), a **backup** that roughly
  doubles when the five die (L7H11), and **dependent consumers** downstream that lose a
  chunk of their score (L9H9: 0.43 → 0.30; L10H1: 0.50 → 0.26).
- A 5-seed check killed one claimed backup (L8H1) and shrank another's effect from 3× to
  2× (L7H11). Single-seed screening flatters its winners; the averaged numbers are the
  ones reported here.

## 1. Background and question

An induction head implements the pattern [A][B] … [A] → [B]: when the context contains a
repeated prefix, the head attends from the current token back to what followed the same
token last time, and pushes that continuation
([Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)).
In GPT-2 small, the commonly identified five are L5H1, L5H5, L6H9, L7H2, L7H10.

The question here is blunt: what happens to induction when you kill all five at once? If
they are *the* induction circuit, repeated-sequence prediction should collapse toward the
no-context baseline. If it doesn't collapse, someone else is doing induction, and the
interesting question becomes who.

## 2. Methods

Everything is GPT-2 small via TransformerLens, on Colab.

- **Input**: [BOS] + 20 random tokens, repeated twice. On the second copy, a model with
  working induction can predict every token by copying from the first copy.
- **Induction score** (per head): attention from each second-copy position back to the
  token 19 positions earlier (the token after the same token's first occurrence), averaged
  over positions. This is an attention-pattern measure — it says where a head looks, not
  what it causally contributes.
- **Behavior metric**: mean loss on the second copy. Clean model: 1.098. First-copy loss —
  random tokens with no repetition to exploit — is 12.362, which I treat as the ceiling:
  the loss of a model doing no induction at all.
- **Ablation**: zero the head's output via its `hook_z`.
- **Control**: L3H0, a head with negligible induction score, ablated the same way.
- **Seeds**: 0–4 (five random token sequences); I report means and the max−min spread.

**Reproducibility.** The numbers in this post come from the original Colab run. A
self-contained reimplementation of the protocol lives in this site's repo at
[`assets/code/2026-07-29-induction-repro.py`](https://github.com/OE-GOD/OE-GOD.github.io/blob/main/assets/code/2026-07-29-induction-repro.py)
(commit `ad6bb66`). Rerunning it with fresh random token streams reproduces every
qualitative claim: the same top five heads, a joint ablation landing about a quarter of
the way to ceiling, L5H0's score identical to the fourth decimal under ablation, both
downstream consumers dropping, and L7H11 roughly doubling. The exact loss values shift
with the token stream (clean 1.00 vs 1.10, ceiling 14.1 vs 12.4) — as they should. In the
rerun, L8H1 again shows a small rise with clean and ablated spreads touching: consistent
with the call below to drop it as unresolvable at five seeds.

## 3. Result 1 — replication

The top five induction scorers are L5H5 (0.88), L7H10 (0.86), L5H1 (0.85), L6H9 (0.82),
L7H2 (0.76) — exactly the five heads the literature names, in slightly different order.
This is a calibration result: it says the score and the setup are measuring the same thing
everyone else measures.

## 4. Result 2 — single ablation needs a control

Killing L5H5 alone moves second-copy loss from 1.098 to 1.544 (+0.446). That looks like a
clean induction effect until you kill the control head: L3H0, which does no induction,
still moves loss to 1.301 (+0.203).

Measured: +0.446 for the top induction head, +0.203 for a non-induction head. Interpreted:
zero-ablation is violent, and roughly half of a single head's apparent effect here is the
generic damage of zeroing *any* working component. Single-head ablation numbers without a
control overstate specificity.

## 5. Result 3 — joint ablation, and what survives

Killing all five together: loss goes to 3.863.

Two readings of that number, one in each direction. Against additivity: the naive sum of
the five single-ablation effects predicts about 2.25, so the joint effect (+2.77) is
superadditive — the five heads partially cover for each other, and killing one lets the
others mask the loss.

Against "these five are the circuit": the ceiling is 12.362. The ablated model's 3.863 is
only about a quarter of the way from clean (1.098) to no-induction (12.362). On the second
copy the model still predicts far, far better than a model without context. Most of the
induction-attributable ability survives the death of all five famous heads. Someone else
is doing it.

## 6. Result 4 — who's doing it: three species

I looked at the induction scores of every other head, clean versus all-five-ablated,
across the five seeds. Three distinct behaviors show up:

- **Independent** — L5H0: 0.49 → 0.49, identical in every seed. This is a long-tail
  inductor that simply doesn't react, and mechanically it can't: it sits in the same layer
  as two of the ablated heads, and same-layer heads run in parallel on the previous
  layer's residual stream. Nothing it reads is touched by the ablation. Its perfect
  non-response doubles as a sanity check that the ablation code does what I think.
- **Backup** — L7H11: its induction score roughly doubles when the five die. On the
  single screening seed it looked like 3×; averaged over five seeds it is 2×. The
  flattering first number is the winner's curse — you found the head *because* it looked
  extreme, so its honest effect is smaller. 2× is the number I stand behind.
- **Dependent consumers** — L9H9: 0.43 → 0.30, and L10H1: 0.50 → 0.26. These sit
  downstream of the famous five and lose a chunk of their induction-pattern attention when
  the five die. Interpreted: part of their behavior consumes the earlier heads' output
  rather than computing induction independently.

And one non-result, reported because the method demands it: **L8H1** (0.45 → 0.51) looked
like a second backup on the screening seed. Across five seeds the clean and ablated
spreads overlap; the rise is indistinguishable from noise. Dropped.

## 7. Limitations

- One model, GPT-2 small. The species structure may or may not exist elsewhere.
- Zero-ablation only. It is the most violent intervention — it takes activations off the
  data manifold — and the control head shows how much generic damage it does.
  Mean-ablation is the obvious gentler follow-up and I have not run it.
- The induction score is an attention-pattern measure. It shows where heads look, not
  their causal contribution to the logits; a head can attend inductively and write
  nothing useful.
- Five seeds, and I report max−min spread, not a standard deviation. Claims that survived
  are the ones whose spreads don't overlap; that is a coarse filter.
- One sequence length (20 tokens repeated). Induction at other scales untested.

Suggestive, not settled: the three-species picture is what five seeds and one model
support. Any of the open questions below could complicate it.

## 8. Open questions

- Kill the top 10, 15, 20 induction scorers. Does loss actually climb to the 12.36
  ceiling, or does a stubborn floor remain that attention-pattern scores don't predict?
- Kill L7H11 along with the five. Does a *second* backup rise — is the redundancy one
  layer deep or many?
- Does the same independent / backup / dependent structure appear in larger models, where
  the induction-head population is bigger?

## Acknowledgments / tools

TransformerLens, GPT-2 small, Google Colab. Analysis and write-up developed in
conversation with Claude (Anthropic).
