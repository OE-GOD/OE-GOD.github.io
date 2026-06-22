---
layout: post
title: "Drivers and thermometers: what a model represents vs what it thinks with"
date: 2026-06-22
---

# Drivers and thermometers: what a model represents vs what it thinks with

*This is the unifying post. A long arc — are SAE features real, can you trust a detector out-of-distribution, does the model know more than it says, does that hold for facts — kept circling one distinction without naming it. Reasoning from first principles (with a collaborator pushing me to), it finally fell out in one sentence: **the model reasons *with* concepts but only *carries* facts.** Read-direction = write-direction for a concept; read ≠ write for a fact. Everything else in the arc is a consequence of that.*

*June 2026 · [Code](https://github.com/OE-GOD/sae-feature-realness) · Gemma-2-2b · steering + probing, reproduced*

---

## From first principles

Strip the model to what's mechanically true: it's a vector being transformed, layer by layer, and at the end the vector is turned into the next token. There is no separate reasoning module. **The transformed vector *is* the thinking.** So "what is the model thinking" can only mean: *what's in the vector, and what each step does to it.*

The field's central bet is that a concept is a **direction** in that vector space — and there's a reason it has to be. The model reads its own vectors with dot products (every layer multiplies by a weight matrix). A dot product asks exactly one thing: *how far does this vector point along that direction?* So a direction is the only shape a concept can take that the next layer can cheaply *use*. That's why probes (a dot product), SAE features (directions), and steering (add a direction) work at all.

But "a concept is a direction" makes a strict, testable prediction, and we never turned it on ourselves: **a real direction should both *read* and *write*.** You should be able to decode the concept from the vector (read), *and* add the direction to change the model's behavior (write) — and for a genuine, used concept, it should be the **same direction**. If a concept is readable but not writable, the model *represents* it but doesn't *compute with* it.

## The test, and the result

**Dose-response.** Take the sentiment direction (difference-of-means of the residual stream between positive and negative text) and use the *read* direction as a *write* direction: add `α · direction` and measure the model's sentiment output. If sentiment is a clean linear feature, you get a volume knob.

You do. Steering strength α from −3 to +3 moves the sentiment logit smoothly and monotonically (≈ −0.8 → +2.1, a swing of ~2.9), while a random direction of equal size stays dead flat. A clean, specific knob. Then I ran the *same* protocol on a **truth** direction (true vs false factual claims): the knob barely turns — a swing of ~0.4, ≈7× weaker, nearly flat.

**The controlled version (same direction, read and write, across layers):**

| | read (AUROC) | write (steering swing) |
|---|---:|---:|
| **Sentiment** (concept) | 0.64 – 0.81 | **1.4 – 2.9** |
| **Truth** (fact) | 0.62 – 0.70 | **≈ 0**  (−0.09 … 0.12; one 0.90 blip at L12) |

This controls for the thing that would otherwise confound it: **truth reads *just as well* as sentiment** (~0.66 vs ~0.70). It is not less *represented*. Yet sentiment steers strongly at every layer and truth barely moves — same direction, same layers, same mechanism (truth serves as its own control, proving the steering isn't a generic perturbation). For sentiment, read = write. For truth, **read ≠ write**.

## Drivers and thermometers

That's the whole thing, and it has a name from the very first post in this project (the [Pythia SAE work](/2026/06/01/sae-pythia-160m/)):

> A **driver** is a direction the model both encodes *and acts on* (read = write). A **thermometer** is a direction the model encodes but *doesn't act on* (read, not write). **Sentiment is a driver. Truth, at this scale, is a thermometer.**

And this single axis unifies the entire arc. Everything that worked for sentiment and failed for facts is a *consequence* of driver-vs-thermometer:

- **Trust / abstention worked for sentiment** ([here](/2026/06/21/interpretability-beats-confidence/)) because its concept is a driver — the reliable-feature direction genuinely carries the decision.
- **The model "knew more than it said" for sentiment** ([here](/2026/06/22/does-it-already-know/)) — a driver's answer survives interference and can be dug out — but **not for facts**, because a thermometer was never steering the answer to begin with.
- **Confident hallucination follows directly:** if "truth" doesn't drive the output, then being wrong trips no internal "this is false" lever — there is nothing to stop it.
- **The known puzzle that you can *locate* a fact but *editing* it doesn't stick** (ROME/MEMIT; localization ≠ editability, Hase et al.) is the same statement: facts are gauges, not levers.

## What it says about how AI thinks

The deepest takeaway is that **"what the model represents" and "what the model computes with" are two different things, and they come apart.** "What is the AI thinking" has two layers — what's *present* in the vector (readable) and what's *driving* the output (causal) — and a probe only sees the first. So:

> **Reading a model's mind is not enough.** The thought you decode might be one the model isn't using. To know what it's actually *thinking with* — not just *holding* — you have to test what *moves the answer*, not just what you can read.

The model **reasons with concepts** (levers it pulls) and **merely carries facts** (gauges it reads but doesn't act on). That's a concrete, mechanical picture of machine cognition, not a metaphor — and it's why interpretability-by-probing can quietly overclaim.

## Honest scope, and the prediction that matters

- **2B model.** Marks & Tegmark found the truth direction becomes *causal* at 13B+. So the precise claim is: **at small scale, truth is a thermometer.** This makes a sharp, falsifiable prediction — **truth's write-swing should rise with model scale, crossing from thermometer to driver** — which is the real frontier experiment, and needs more than a laptop.
- Mean-pooled directions, few-shot judging, one steering protocol, modest read-AUROC (small model); truth showed a faint write effect at L12 (0.90) — not literally zero, just ~3× weaker than sentiment there and ~0 elsewhere.

Reproductions: [`firstprinciples_steering.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/firstprinciples_steering.py), [`firstprinciples_truth_steering.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/firstprinciples_truth_steering.py), [`firstprinciples_read_vs_write.py`](https://github.com/OE-GOD/sae-feature-realness/blob/main/firstprinciples_read_vs_write.py).

---

The arc started at "are SAE features real?" and ends here: **some of what a model represents, it reasons with; some it just carries — and only steering, not reading, tells you which.** That distinction, and the scale prediction attached to it, is the cleanest thing I know to say about how this model thinks.
