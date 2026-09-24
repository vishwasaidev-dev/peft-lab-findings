---
title: Where does a trained behavior actually live?
layout: default
---

# Where does a trained behavior actually live?

[← back to index](index.html)

## Starting point: a zoo of real adapters

We pulled 7 independently-trained, publicly-hosted LoRA adapters off the Hugging Face Hub, all fine-tuned for the same base model (`Qwen/Qwen2.5-0.5B-Instruct`), by different authors for different tasks — e-commerce field extraction, a memory-consolidation assistant, a generic instruction task, and others. Same base means same architecture, same layer count, so their weight matrices are directly comparable.

A few things fell out immediately from just diffing their weights:

- **The delta-weight norm grows with depth.** In 6 of 7 adapters, the size of the LoRA update at the last transformer layer is 1.5–2x the size at the first layer. Later layers get bigger nudges, consistently, across adapters nobody coordinated with each other.
- **Nobody uses their full rank budget.** Using singular-value entropy as an effective-rank estimate: a rank-8 adapter only really uses about 5.9 dimensions; rank-16 adapters land around 12–14; the one rank-32 adapter uses about 26. Every adapter left capacity on the table.
- **Independent adapters are nearly orthogonal.** Pairwise cosine similarity between adapters' weight-update matrices, same layer, same module, is ~0.00–0.01 — even between three adapters trained by the *same author* on a *related task family* (memory consolidation / retrieval / storage). Different fine-tunes carve unrelated directions through weight-space, even starting from the identical base model.

## One adapter refuses. Where is that coming from?

One of the seven (`domhotdog`, of unknown provenance) flatly refused an ordinary customer-support question — *"I'm sorry, but I can't assist with that"* — while all six others answered normally. That gave us a real, spontaneously-occurring behavior to trace, rather than one we injected ourselves.

### Structural: does it spend its weight budget differently?

We summed each adapter's delta-weight norm per (layer, module) cell and z-scored `domhotdog` against the other six. It's not uniform:

- **Late `mlp.up_proj` / `mlp.down_proj` (layers 13–23): significantly over-weighted**, peaking at `down_proj` layer 22 (z = +5.5).
- **Early `gate_proj` (layers 1–3): significantly under-weighted** (z ≈ −2.3 to −3.5).
- `q_proj`/`k_proj`/`v_proj` mostly unremarkable.

### Functional: does the computation actually diverge?

We ran the model with the adapter on vs. off across three prompts (the refusal-triggering one, a neutral one, and a benign control), capturing the residual stream at every layer. Adapter-on vs. adapter-off divergence is nearly identical across all three prompts through layer ~10 — early layers process tokens the same way regardless of content. From layer ~14 onward, the refund prompt's divergence pulls sharply ahead of the others, roughly doubling by layers 20–23. The "decision" to refuse only crystallizes in the last third of the network.

Both lines of evidence point the same way, and it matches published interpretability findings (e.g. Arditi et al. on refusal being mediated by a fairly localized late-layer direction) — this isn't uniformly-distributed storage, there's a real late-layer concentration.

## Is it actually necessary, or just sufficient?

Correlation isn't causation, so we tested it directly. We identified the 16 weight cells (out of ~150 total) with the strongest structural over-investment, and built two surgical variants of the adapter:

- **`late_only`** — keep *only* those 16 cells, zero everything else.
- **`late_ablated`** — zero *only* those 16 cells, keep everything else.

| Prompt | full `domhotdog` | `late_only` (16/150 cells) | `late_ablated` (16 cells removed) |
|---|---|---|---|
| "What should I do if a customer wants a refund?" | refused | **refused** | **refused** |
| "How do I request a refund from support?" | refused | refused (hedged) | refused |
| "Can you help me process a return?" (never refused originally) | answered | answered | answered |
| Four-bar linkage question (control) | answered | answered | answered |

**Sufficiency confirmed:** 16 cells alone reproduce the refusal.
**Necessity denied:** zeroing those exact 16 cells doesn't kill it — the rest of the network still carries it.

So this isn't a single point of failure. It's **redundantly encoded** — a late-layer subcircuit is *sufficient* to trigger the behavior in isolation, but the behavior is backed up elsewhere too. That's a closer match to cortical redundancy than to a single "refusal neuron."

## Does it cross a language boundary?

We asked the same refund question, unmodified, in English, Spanish, French, Hindi, and Marathi, through `domhotdog` — an adapter trained only on English text.

| Language | Base model | `domhotdog` |
|---|---|---|
| English (training language) | answers | **refuses cleanly** |
| Spanish | answers fluently | answers — no refusal at all |
| French | answers fluently | hedges ("I can help, but I can't perform actions that could be considered...") — partial |
| Hindi | **incoherent** (model doesn't know the language well) | incoherent |
| Marathi | **incoherent** | incoherent |

Two separate things are visible here, and they need to be pulled apart:

1. **Does the base model know the language at all?** For Hindi and Marathi, no — a 0.5B model simply doesn't have the capacity to speak them coherently, with or without any adapter. That's a hard prerequisite that has nothing to do with LoRA.
2. **Given that the base model does know the language, does the trained behavior transfer?** Only where (1) succeeds is this a real test. Result: full transfer to English, partial to French, none to Spanish — despite Spanish and French being roughly equally "close" to English by most naive measures.

The mechanism isn't matching literal tokens (by the time a token reaches the deep layers where this fires, it's already a heavily-processed, largely language-agnostic vector — see Wendler et al. on multilingual models routing through a shared internal "concept space"). But the transfer through that shared space is real and *uneven*, not a clean universal guarantee.

[Next: Teaching and un-teaching guardrails →](02-teaching-and-removing-guardrails.html)
