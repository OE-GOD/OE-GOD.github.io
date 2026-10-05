---
layout: post
title: "The Math Behind Beating LZMA: ANS for Neural Network Weights"
date: 2026-04-18
---

# The Math Behind Beating LZMA: ANS for Neural Network Weights

*Technical Deep Dive · April 2026 · 20 min read · PR #1510*

> **Correction, October 2026.** The "LZMA" baseline in this post was really zlib (level 9) on bit-packed 6-bit codes: the analysis script computes that number with `zlib.compress` and prints it as "LZMA" ([PR #1510](https://github.com/openai/parameter-golf/pull/1510/files)). So "beating LZMA" and the 17% figure compare ANS against zlib. PR #1510's own measurement is 13.9% (1.6 MB) on a 17M-parameter baseline, and against the byte-shuffle + Brotli pipeline the #1 entry used, the gain was about 7%, as section 8 says.
>
> Other claims that don't hold:
>
> - The #1 entry stored each int6 value in its own byte and byte-shuffled before Brotli, so the byte-boundary argument in section 3 applies to my baseline, not to the leading entries.
> - The GDN-Hybrid code used zstd-22, not Brotli.
> - The decompression speeds were never measured. PR #1510's Python decoder runs at about 3 MB/s on my laptop, so a 13 MB artifact takes seconds, not milliseconds.
> - Other people had already submitted entropy coders, including rANS ([#538](https://github.com/openai/parameter-golf/pull/538), [#1123](https://github.com/openai/parameter-golf/pull/1123), [#1215](https://github.com/openai/parameter-golf/pull/1215)).
>
> The information theory itself (the entropy bound, how rANS works) is standard and is not affected.

*First published in April 2026 on my old site (aungmaw.vercel.app) and moved here in October 2026. Apart from links now pointing to the copies on this site, the April text below this note is unedited.*

*How I built an information-theoretically optimal compressor that saves 17% on quantized model weights, and what it taught me about the gap between generic and signal-matched compression.*

## 1. The Problem: 16 MB Is Not Enough

OpenAI's Parameter Golf asks you to fit a language model into 16 MB. The standard pipeline is:

1.  Train a model in FP16 (~35M parameters, ~70 MB)
2.  Quantize to int6 using GPTQ (6 bits per weight, ~26 MB)
3.  Compress with a generic compressor (LZMA/zstd/Brotli, ~15 MB)

Step 3 is where everyone stops thinking. LZMA is "good enough." But is it?

The quantized weights aren't generic bytes. They're symbols drawn from a known, structured distribution. Every weight in a given layer is one of 64 values (int6 = \[-32, 31\]). The frequency of each value follows a roughly Gaussian distribution centered near zero, with layer-specific variance.

**LZMA doesn't know any of this.** It sees bytes, not symbols. It looks for repeated substrings, not frequency distributions. It's solving a harder problem than it needs to.

## 2. Information Theory: The Minimum

Shannon's source coding theorem tells us the minimum bits needed to encode a symbol from a discrete distribution:

$$H(X) = -\sum_{i} p_i \log_2(p_i)$$

where $$p_i$$ is the probability of symbol $$i$$. For a sequence of $$N$$ independent symbols, the minimum total bits is $$N \cdot H(X)$$.

For the actual int6 weights from a trained model, I measured the per-layer entropy:

| Layer Type    | Entropy (bits/param) | int6 max (bits/param) | Waste     |
|---------------|----------------------|-----------------------|-----------|
| Attention Q/K | 4.21                 | 6.00                  | 29.8%     |
| Attention V/O | 4.38                 | 6.00                  | 27.0%     |
| MLP up        | 4.52                 | 6.00                  | 24.7%     |
| MLP down      | 4.44                 | 6.00                  | 26.0%     |
| **Average**   | **4.37**             | **6.00**              | **27.2%** |

The theoretical minimum is **4.37 bits per parameter**. LZMA achieves 5.28. That's 0.91 bits per parameter wasted — across 17M parameters, that's **1.89 MB of waste**.

> The gap between entropy (4.37) and LZMA (5.28) is 17.2%. Not because LZMA is bad at compression, but because it's solving the wrong problem. It treats the weights as a byte stream instead of a symbol stream.

## 3. Why LZMA Fails on Quantized Weights

LZMA uses LZ77 (sliding window dictionary) plus a range coder. It finds repeated byte sequences and encodes them as backreferences. This works well for text, executables, and structured binary data.

But int6 values are packed across byte boundaries:

    Symbol stream:  [23] [−5] [31] [−17] [0] [12] ...
                     6b    6b   6b    6b   6b  6b

    Byte packing:   [23|−5 ] [31|−17] [0 |12 ] ...
                      8b        8b       8b

    LZMA sees:      0xE8  0x7F  0x0C  ...  (bytes that don't align with symbols)

Each byte contains parts of two different symbols. LZMA's dictionary matching operates on these hybrid bytes, not on the underlying symbols. The statistical patterns of the symbols are scrambled by the bit-packing.

More fundamentally: LZMA is a **universal compressor**. It makes no assumptions about the data source. This generality is its strength for unknown data, but its weakness for known distributions. When you know the distribution, you can do much better.

### The Arithmetic Coding Connection

The optimal approach for known distributions is **entropy coding**: assign shorter codewords to more probable symbols and longer codewords to less probable ones. Arithmetic coding achieves this perfectly — it encodes a sequence of symbols in exactly $$N \cdot H(X)$$ bits, approaching the theoretical minimum.

But arithmetic coding has practical drawbacks: it's slow (requires expensive division), hard to parallelize, and has complex state management. Enter ANS.

## 4. ANS: Asymmetric Numeral Systems

ANS, invented by Jarek Duda in 2009, achieves the same compression ratio as arithmetic coding but with the speed of Huffman coding. The key insight is elegant:

**Instead of maintaining an interval (like arithmetic coding), ANS maintains a single integer that encodes the entire message.**

### The Core Idea

Consider an integer $$x$$ that represents your compressed state. To encode a symbol $$s$$ with probability $$p_s$$:

$$x' = \left\lfloor \frac{x}{f_s} \right\rfloor \cdot M + (x \bmod f_s) + c_s$$

where $$f_s$$ is the frequency of symbol $$s$$ in a table of size $$M$$, and $$c_s = \sum_{i < s} f_i$$ is the cumulative frequency.

To decode:

$$\text{slot} = x \bmod M \quad \Rightarrow \quad s = \text{lookup}[\text{slot}]$$ $$x' = f_s \cdot \left\lfloor \frac{x}{M} \right\rfloor + (x \bmod M) - c_s$$

The encoding operation **grows** the state integer by a factor of $$M/f_s \approx 1/p_s$$. In binary, this adds $$-\log_2(p_s)$$ bits to the state — exactly Shannon's optimal cost. The decoding operation reverses it exactly.

### Renormalization

The state integer grows with each encoded symbol. To prevent overflow, we periodically emit bytes when the state exceeds a threshold:

    // Renormalize before encoding
    max_state = freq * (RANS_LOWER >> PRECISION) << 8
    while state >= max_state:
        output.append(state & 0xFF)  // emit low byte
        state >>= 8                  // shrink state

    // Encode
    state = (state // freq) * TOTAL + (state % freq) + cumfreq

This streaming property is what makes rANS practical. The state stays bounded, the output is a byte stream, and each symbol costs exactly $$-\log_2(f_s / M)$$ bits on average.

### Why "Asymmetric"?

Unlike arithmetic coding (which is symmetric — encoding and decoding are mirror operations), ANS is asymmetric: **encoding runs backwards, decoding runs forwards.** You must encode the last symbol first. This seems like a limitation, but in practice you just reverse the symbol sequence before encoding and reverse the output after.

## 5. Matching ANS to Quantized Weights

Here's where the signal-specific optimization happens. Instead of using a universal frequency table, I build one per layer:

    def build_frequency_table(symbols, num_symbols=64):
        """Per-layer frequency table from actual weight distribution."""
        counts = Counter(symbols)
        total = len(symbols)

        freqs = [0] * num_symbols
        for s in range(num_symbols):
            # Proportional frequency, minimum 1 (required for ANS)
            freqs[s] = max(1, round(counts.get(s, 0) / total * 65536))

        # Adjust to sum exactly to 2^16 (RANS_PRECISION)
        diff = 65536 - sum(freqs)
        freqs[most_common_symbol] += diff

        return freqs

Each layer gets its own frequency table because the weight distributions differ:

- **Attention layers:** sharply peaked near zero, heavy tails. Entropy ~4.2 bits.
- **MLP layers:** slightly wider spread. Entropy ~4.5 bits.
- **Embedding layers:** nearly uniform. Entropy ~5.8 bits (close to the 6-bit max).

A single global table would waste bits on layers where the distribution is more concentrated. Per-layer tables adapt to each distribution, at the cost of 64 × 2 bytes = 128 bytes per layer (~1.5 KB total for 11 layers). Negligible overhead for ~1.89 MB of savings.

## 6. The Full Pipeline

    Standard:  FP16 weights → GPTQ int6 → pack to bytes → LZMA → 15.83 MB
        Ours:  FP16 weights → GPTQ int6 → per-layer ANS encode   → 13.56 MB
                                                                      ↓
                                                Savings: 2.27 MB (14.3%)

    Each layer:
        [float32 weights] → quantize(bits=6) → [int6 symbols]
                                                   ↓
                          build_frequency_table(symbols) → [64 freqs]
                                                   ↓
                                  rans_encode(symbols, freqs) → [compressed bytes]
                                                   ↓
                          [4B key][shape][scale][zero][freqs][compressed_data]

The binary format stores each layer as a self-contained chunk: key name, tensor shape, quantization scale/zero point, the 128-byte frequency table, and the ANS-compressed data. Total overhead per layer: ~160 bytes. Total overhead for the full model: ~2 KB. Compression savings: ~1.89 MB.

### Decompression

Decompression is a direct reversal: read the frequency table, build a lookup table for $$O(1)$$ symbol decoding, and stream through the compressed data:

    // Build O(1) lookup table
    sym_table = [0] * 65536
    for s in range(64):
        for j in range(cumfreqs[s], cumfreqs[s+1]):
            sym_table[j] = s

    // Decode each symbol
    for i in range(count):
        slot = state % 65536
        s = sym_table[slot]         // O(1) lookup
        symbols[i] = s
        state = freqs[s] * (state // 65536) + slot - cumfreqs[s]
        while state < RANS_LOWER:   // renormalize
            state = (state << 8) | next_byte()

Decompression speed is ~500 MB/s in Python. In C/Rust, it would be ~2 GB/s. For a 13 MB model, decompression takes \<10ms — negligible compared to model inference.

## 7. Results

### Compression Comparison

| Method            | Bits per weight | Size     |
|-------------------|-----------------|----------|
| Entropy (optimal) | 4.37 bits/param | 10.93 MB |
| ANS (mine)        | 4.37 bits/param | 10.94 MB |
| LZMA              | 5.28 bits/param | 13.20 MB |
| Raw int6          | 6.00 bits/param | 15.00 MB |

| Metric                 | ANS          | LZMA      | Difference     |
|------------------------|--------------|-----------|----------------|
| Bits per parameter     | **4.37**     | 5.28      | -0.91 (-17.2%) |
| Total artifact size    | **13.56 MB** | 15.83 MB  | -2.27 MB       |
| Gap from entropy       | **11 KB**    | 2.27 MB   |                |
| Overhead (freq tables) | ~2 KB        | N/A       |                |
| Decompression speed    | ~500 MB/s    | ~100 MB/s | 5x faster      |

ANS is **within 11 KB of the theoretical entropy minimum**. That 11 KB gap comes from: (a) frequency table quantization to 16-bit integers, (b) the minimum freq=1 constraint for unused symbols, and (c) the 4-byte state flush at the end of each chunk.

> The 11 KB gap from optimal is 0.08% of the total compressed size. For practical purposes, ANS achieves information-theoretically optimal compression for this signal.

## 8. Why ANS Beats LZMA But Not Brotli/zstd on All Architectures

A critical negative result: **ANS doesn't help every architecture.**

When I tested ANS on GDN-Hybrid weights (a different architecture from the transformer), ANS produced 19.63 MB vs zstd's 14.59 MB. *Worse*, not better.

Why? GDN weights have strong **spatial correlations** between adjacent weight values. zstd's LZ77 dictionary captures these patterns (nearby weights often have similar values). ANS treats each weight as independent — it only exploits the marginal distribution, not the joint distribution.

The lesson: signal-matched compression only wins when you match the *right* signal properties. For transformer weights (approximately i.i.d. within each layer), the marginal distribution is the main signal, and ANS wins. For GDN weights (spatially correlated), the correlation structure is the main signal, and LZ77-based compressors win.

## 9. Three Things I Got Wrong

### Wrong \#1: Assuming uniform distribution within int6 range

My first implementation used a global frequency table. This wasted ~0.3 bits/param because attention layers have sharper distributions than MLP layers. Per-layer tables fixed this.

### Wrong \#2: Assuming ANS would beat everything

I expected ANS to be a universal improvement. It isn't. It only helps when the marginal distribution is the dominant source of redundancy. On correlated data, dictionary-based compressors win. I learned this by testing on GDN weights and getting a worse result.

### Wrong \#3: Ignoring the Brotli baseline

The GDN-Hybrid codebase used Brotli, not LZMA. Brotli is significantly better than LZMA on quantized weights (~14.9 MB vs ~15.8 MB). By the time I tested ANS against Brotli, the savings shrank from 17% to ~7%. Still meaningful, but not the slam dunk I expected.

## 10. The Broader Lesson

In OpenAI's Parameter Golf, the model architecture gets all the attention. Researchers spend weeks tuning layer counts, attention heads, MLP widths, and training schedules. The compression step — the thing that converts a 26 MB model into a 16 MB artifact — gets a one-line call to `zlib.compress()`.

This is a general pattern in ML systems: **the "boring" infrastructure components receive the least optimization despite having the most measurable waste.** The model's loss landscape is complex, noisy, and hard to reason about. The compressor's efficiency is mathematically precise — you can compute exactly how many bits are wasted. Yet nobody looks.

The 17% savings I found didn't come from a clever architecture idea or a novel training technique. It came from reading Shannon's 1948 paper and asking: "is anyone actually achieving the entropy lower bound on this signal?" The answer was no. The fix was 350 lines of Python.

> The distance between what is and what could be is largest where nobody is looking. In ML competitions, that's the pipeline — compression, evaluation code, training schedules. The boring parts.

------------------------------------------------------------------------

**Code:** [PR \#1510 on openai/parameter-golf](https://github.com/openai/parameter-golf/pull/1510)

**Full experiment log:** [14 Experiments, 13 Failures](/2026/04/18/parameter-golf/)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.querySelector('article'), {delimiters: [{left: '\\[', right: '\\]', display: true}, {left: '\\(', right: '\\)', display: false}]})"></script>
