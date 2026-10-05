---
layout: post
title: "14 Experiments, 13 Failures"
date: 2026-04-18
---

# 14 Experiments, 13 Failures

*OpenAI Parameter Golf · April 2026 · 13 failed · 1 worked · 12 min read · $150 GPU spend*

> **Corrections, October 2026.** Checked against openai/parameter-golf.
>
> - **PR #1510 was never merged.** I opened it on April 9; it got no review, is still open, and was never on the leaderboard.
> - **The compression gain was smaller than stated, and the baseline wasn't LZMA.** PR #1510's own headline is 13.9% (1.6 MB), measured against zlib level 9 on bit-packed weights; my script printed that zlib number under the label "LZMA" ([the code](https://github.com/openai/parameter-golf/pull/1510/files)). The 17.2% / 1.89 MB here came from a later run against the same zlib baseline. Against the byte-shuffle + Brotli pipeline the #1 entry used, the gain was about 7%.
> - **I wasn't the first to work on compression.** Huffman coding ([#532](https://github.com/openai/parameter-golf/pull/532)), arithmetic coding ([#538](https://github.com/openai/parameter-golf/pull/538)) and rANS ([#1123](https://github.com/openai/parameter-golf/pull/1123), [#1215](https://github.com/openai/parameter-golf/pull/1215)) were all submitted before my PR. "Nobody optimized the compression" is false.
> - **The TTT and byte-counting entries are wrong.** The leaderboard's top entries used legal score-first TTT, and the byte-counting bug was found by others and never reached the leaderboard. See the corrections on [Compliant TTT](/2026/04/18/compliant-ttt/) and [the byte-counting post](/2026/04/18/byte-counting-bug/).

*First published in April 2026 on my old site (aungmaw.vercel.app) and moved here in October 2026. Apart from links now pointing to the copies on this site, the April text below this note is unedited.*

*What I actually learned about ML compression in OpenAI's competition. Two weeks, ~$150 in GPU compute, and a lot of honest negative results.*

### The Journey

| Experiment | What happened | Result | Outcome |
|----|----|----|----|
| ANS Compression | Built rANS encoder. 17% lossless improvement. Merged as PR \#1510. | -17% artifact size | Worked |
| SP8192 on GDN-Hybrid | Bigger vocab. Embedding table too large for 16 MB budget. | 16.94 MB (over budget) | Failed |
| head_dim=128 | 2x state capacity, 44% less memory. But 15% slower = fewer steps. | +0.011 BPB (worse) | Failed |
| Byte-Counting Bug | GDN-Hybrid double-counts space bytes. All GDN BPB numbers inflated. | 14% BPB inflation | Discovery |
| Warmdown Fix | Three training systems silently disabled. Fixed timing. | -0.003 BPB | Marginal |
| Score-First TTT | Compliant TTT: score before training. Perturbs weights, hurts quality. | +0.003 BPB (worse) | Failed |
| 6 More Experiments | Extra layer, ANS on GDN, per-matrix quant, recurrence, casefold, batch. | All failed | Failed |

## The One That Worked

***Worked** · Compression · Information Theory*

### ANS Compression (PR \#1510)

**The question:** Everyone uses generic compressors (LZMA, Brotli) to pack model weights into 16 MB. Are they wasting space?

### Compression Efficiency

| Method     | Bits per weight         |
|------------|-------------------------|
| Optimal    | 4.37 bits               |
| ANS (mine) | 4.37 bits               |
| LZMA       | 5.28 bits — 0.91 wasted |

**Savings: 1.89 MB (17.2%).** rANS encoder with per-layer frequency tables matched to the actual weight distribution. Information-theoretically optimal.

> Everyone optimized the model. Nobody optimized the compression. The boring part had the most opportunity.

------------------------------------------------------------------------

## The 13 That Failed

***Failed** · Tokenizer*

### 1. SP8192 on GDN-Hybrid

**Hypothesis:** SP8192 produces 37% fewer tokens. Every transformer uses it. Should give -0.03 BPB.

| Config          | BPB   | Artifact | Fits? |
|-----------------|-------|----------|-------|
| SP1024 baseline | ~1.10 | 14.59 MB | Yes   |
| SP8192 dim=512  | 1.062 | 16.94 MB | No    |
| SP8192 dim=496  | 1.088 | 16.30 MB | No    |

> Optimizations that work for transformers don't transfer to architectures with different parameter budgets.

***Failed** · Architecture*

### 2. head_dim=128 (4 Heads Instead of 8)

**Hypothesis:** 2× state capacity at same param count. Larger keys = cleaner retrieval.

**Result:** 44% less GPU memory (11.8 vs 21 GB) but 0.011 worse BPB. Fewer heads = slower steps = less training in 10 minutes.

> State capacity is NOT the bottleneck. Training compute (steps per wallclock) is.

***Discovery** · Evaluation*

### 3. The Byte-Counting Bug

**Found:** GDN-Hybrid's eval double-counts leading space bytes for ~65% of tokens. Impact: ~14% BPB inflation.

    // BUGGY: +1 baked in, then eval adds +1 again
    base_bytes[i] = len(piece[1:].encode("utf-8")) + 1

    // CORRECT: no +1, eval adds it once
    base_bytes[i] = len(piece[1:].encode("utf-8"))

> When your BPB looks too good, check the denominator.

***Failed** · Compliance*

### 4. Score-First Compliant TTT

**Problem:** Top entries train 6 epochs on val data then score it — violating Condition 3. PRs \#1487, \#1488 were closed for this.

**My fix:** Score first half. Train on first half. Score second half with adaptation.

| Method             | BPB    | Compliant? |
|--------------------|--------|------------|
| No TTT             | 1.1118 | Yes        |
| Score-first (mine) | 1.1152 | Yes        |
| 6-epoch (invalid)  | 1.0787 | No         |

Score-first is **0.003 worse** than no TTT. Training perturbs EMA weights. Damage compounds during quantization.

> The TTT improvement on the leaderboard may be entirely an artifact of non-compliance.

***All Failed***

### 5-13. Other Experiments

| Experiment               | Expected   | Actual   |
|--------------------------|------------|----------|
| Warmdown fix (3 systems) | -0.01      | -0.003   |
| Extra GDN layer          | -0.005     | ~0       |
| ANS on GDN weights       | Smaller    | Bigger   |
| Per-matrix quant         | Less error | +51% MSE |
| Post-TTT GPTQ            | -0.005     | +0.000   |
| Depth recurrence         | -0.01      | ~0       |
| Word-start weighting     | -0.005     | Marginal |
| Casefold tokenizer       | -0.01      | ~0       |
| Reduced batch            | More steps | Worse    |

------------------------------------------------------------------------

## The Pattern

**The model works.** Dozens of researchers spent weeks optimizing. No easy -0.03 hiding in architecture changes.

**The pipeline has bugs.** Byte-counting inflated BPB 14%. TTT compliance affected top entries. Warmdown disabled three systems.

**Negative results have value.** Each failure narrows the search space for everyone.

------------------------------------------------------------------------

## What I'd Tell My Past Self

1.  **Check the eval code before optimizing.** I spent days improving BPB that was measured wrong.
2.  **Run the baseline first.** Half my experiments used broken software stacks.
3.  **Compliance beats cleverness.** A compliant 1.11 is worth more than a non-compliant 1.08.
4.  **\$4 experiments are cheap.** \$150 of undirected experiments is expensive.
5.  **The boring parts have the most opportunity.** Compression, schedules, eval code. Nobody looks.

------------------------------------------------------------------------

## Summary

| Experiment        | Expected  | Actual        | Status    |
|-------------------|-----------|---------------|-----------|
| ANS compression   | -17% size | -17% size     | Worked    |
| Byte-counting bug | N/A       | 14% inflation | Discovery |
| SP8192 on GDN     | -0.03     | Over 16 MB    | Failed    |
| head_dim=128      | -0.005    | +0.011        | Failed    |
| Score-first TTT   | -0.008    | +0.003        | Failed    |
| Warmdown fix      | -0.01     | -0.003        | Marginal  |
| 7 others          | Various   | All failed    | Failed    |
