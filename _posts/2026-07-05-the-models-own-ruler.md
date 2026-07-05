---
layout: post
title: "The model's own ruler: I preregistered the last escape hatch, and the instrument failure was the finding"
date: 2026-07-05
---

# The model's own ruler

*Fifth post in the SAE-geometry series. Four posts of negatives left exactly one
objection standing: maybe I measured geometry with the wrong ruler. This post is the
test of that objection — run under full pre-registration, with the hypothesis, the
gates, the decision lines, and my honest predictions frozen in a git commit before
any data existed. It's also a post about what to do when your experiment returns
neither yes nor no.*

*July 2026 · [Code](https://github.com/OE-GOD/feature-geometry-locality) —
`PREREG_ruler.md` and `ruler_test.py`; registration commit `3db734f`, amendment
`fd928ca`, final record `3ff278e`*

---

## The objection

Every negative in this series measured geometry the same way: cosine between decoder
directions. But the model doesn't read its residual stream with cosine — it reads it
through its downstream weights. Two directions at a wide angle could be processed
nearly identically; two nearby directions could be processed differently. So the last
escape hatch: **measure distance the way the model does.** Inject each feature's
direction into the residual stream on real text, let the remaining layers run, and
record the response. If the model's *reactions* to two features are similar, the
model treats them as similar — that's geometry in the model's own coordinates,
measured rather than assumed.

The hypothesis (H1): similarity under the model's own ruler predicts genuine function
(co-firing on held-out text), beyond a matched null and beyond raw cosine.

## Why preregister

Because this series has taught me exactly how I fool myself. With ~80,000 feature
pairs and a dozen defensible analysis choices, *something* will always look
significant if I choose after peeking. So before any measurement, I froze in a
commit (`3db734f`): the exact statistic, the null, three instrument gates, the
decision lines (confirm needs z ≥ +4 on both channels; refute if either ≤ +2 with
gates passing), and a prediction table — including a named "this row is most likely
to break" flag. The commit hash is the registration; nothing could quietly move
afterward.

## The design, briefly

- **Ruler:** inject alpha·W_dec[i] at layer 8, last position, on real contexts;
  propagate through the remaining blocks; the response is the change in the final
  residual. (The injection harness reproduces the clean forward exactly at alpha=0 —
  that's gate G0, and it passed at 1.5e-05.)
- **Function:** co-firing, computed on *disjoint* text from the ruler's contexts.
- **The null:** random rigid rotations of the entire decoder set before injection.
  Rotation preserves every pairwise angle among the 400 directions — all 79,800
  cosines, exactly — and destroys only their *alignment* to the model's weights. So
  anything explainable by "similar directions get treated similarly by any smooth
  map" survives in the null and cancels. If the real decoder beats the rotated
  decoys, the credit can only go to alignment. That's precisely H1.

## What happened (measured)

**Run 1** (32 contexts, 8 rotations): the harness gate and sanity gate passed; the
split-half stability gate **failed** at 0.292 against the frozen 0.3 bar — the exact
row my prediction table flagged as most likely to break. The substantive numbers sat
between the frozen lines: z = +3.73 raw, +2.32 after controlling cosine. Frozen
verdict: **indeterminate — instrument**.

Per protocol I did not tweak and retry. I registered an amendment (`fd928ca`) *before*
rerunning: 4× the contexts, 12 rotations, everything else identical — with a new
frozen fork: *if run 1's signal was sampling noise, more data will push stability up
(~0.55 by Spearman-Brown) and z toward 0.*

**Run 2** (128 contexts, 12 rotations): stability went **down** (0.258). The
deflated responses concentrated further onto a single axis (top-1 energy 0.34 →
0.78): a per-feature scalar times one shared pattern. And z fell to **+1.44** raw,
**+0.89** partial — the noise branch, exactly as registered. Frozen verdict:
**indeterminate — instrument**, again.

Prediction scorecard, failures first: G1 stability *failed both runs* (I predicted
pass at 0.5–0.85, then 0.45–0.65 — wrong twice). What held: the pre-flagged
most-likely-to-break row was the one that broke; the z range (+1..+5); and the
amendment's noise branch called run 2's collapse before it ran.

## What it means (interpreted — my reading, marked as such)

Formally this is INDETERMINATE, not a refutation: the gates block a clean kill, and I
report it that way.

But the *reason* the gates fail is itself informative, and it's the finding I take
away. Stability that *drops* when you quadruple the data is not noise you can
average away — it's the signature of measuring something that isn't stably there.
What the model's response to a feature direction actually contains, on this
instrument: one big feature-agnostic component (the generic reaction), one
per-feature *magnitude* scalar riding a shared pattern, and a feature-specific
residue too unstable to survive a split-half check. The model's own ruler barely
distinguishes its own features.

Which closes the escape hatch in an unexpected way. The objection was "you measured
geometry in the wrong coordinates." The answer isn't "I re-measured and it's still
negative." It's: **in the model's own coordinates, the features are still nearly
interchangeable — there is not enough feature-specific structure in the responses to
build any functional map from, in any coordinates derived from them.** And even
taking the unstable whisper at face value, its ceiling is a rank correlation of
~0.011 — one ten-thousandth of the pair variance. Not a map.

## Caveats

- GPT-2-small only, one layer (8), one injection dose (~3% of residual norm), one
  function measure (co-firing). Any of these could in principle hide structure this
  instrument misses.
- INDETERMINATE is the formal verdict; "the responses barely encode feature
  identity" is my interpretation of *why* the instrument can't be stabilized, not a
  gate-passing result.
- What would overturn this: an instrument built from model responses that passes a
  split-half reliability gate and beats rotation nulls on an independent function
  measure. I registered no further amendments because the failure got *worse* with
  more data — but a genuinely different response readout (e.g., per-head attention
  patterns rather than final-residual shifts) is untested.

## Coda

Suggestive, not settled — but the suggestion is pointed. Four posts ago I hoped
geometry was a map. Then I hoped the model's warping of it was the map. The
preregistered answer: the model barely reads individual features at all on this
axis, and no re-metrization can conjure information that the responses don't carry.

The part I'd defend hardest isn't the result — it's the receipts. The failure mode
was named in writing before the experiment ran; the rerun's outcome was predicted in
a committed amendment before it ran; and both wrong predictions of mine are printed
above, next to the ones that held. In a field where it's easy to publish a beautiful
number, I'd rather build the track record of calling my own shots — including the
misses.
