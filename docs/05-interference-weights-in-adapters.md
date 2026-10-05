---
title: Does "interference weights" explain the leak?
layout: default
---

# Does "interference weights" explain the leak?

[← back: What this adds up to](04-synthesis.html)

[Anthropic's "Characterizing interference weights in a tiny language model"](https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html) (Turner, Wu, Batson, 2026) gives a precise, causal way to ask a question we only answered behaviorally on the previous pages: is a weight actually doing anything, and is what it's doing good or bad? They define two measurements on a 1-layer toy transformer's weights:

- **Effectiveness** — a cheap, Fisher-information estimate of how much a weight's removal would move the model's output distribution (a 2nd-order approximation of KL divergence).
- **Helpfulness** — the expensive ground truth: ablate the weight, see whether loss goes up (the weight was helping) or down (the weight was harmful).

Their headline finding: a model's *most effective* weights are overwhelmingly helpful, and harmful weights stay confined to a lower-effectiveness band — but even so, the model remains "dense" in that basis; no clean threshold isolates a fully sparse, fully-interpretable circuit.

We asked: does this explain our own `banana_redefined` leak from the [previous page](02-teaching-and-removing-guardrails.html) — v1 leaking refusal into apple pie and refund questions, v2 (same task, plus contrastive negatives) not leaking?

## What we measured

A LoRA adapter is already a weight decomposition — `ΔW = B·A`, spread across 96 cells (24 layers × 4 target modules). That's a much smaller and more concrete object than the paper's full virtual-weight basis, so rather than reconstructing virtual weights across six path families, we ablate the real, already-existing LoRA cells directly, and measure both metrics at the one token position that matters for this task — the first generated token, where "refuse" and "answer normally" diverge.

For each of v1 and v2, we:
1. Ranked all 96 cells by **effectiveness** on the actual 8 training examples (cheap — this is the step the paper says scales).
2. Took the top 10 most-effective cells and measured **helpfulness** on three disjoint prompt sets: `TRAIN` (the banana examples actually trained on), `ADJACENT` (other fruit/food topics, never trained), `UNRELATED` (refunds, pets, Wi-Fi — zero topical connection to bananas).

(Full script: [`code/13_interference_weights_in_adapter_leaks.py`](https://github.com/vishwasaidev-dev/peft-lab-findings/blob/main/code/13_interference_weights_in_adapter_leaks.py).)

## What we found

| | help(train) | help(adjacent) | help(unrelated) |
|---|---:|---:|---:|
| v1 (leaky) | +0.00016 | −0.03796 | −0.03119 |
| v2 (contrastive) | +0.00035 | −0.02015 | −0.04687 |

(Positive = helpful; negative = harmful, in the paper's sign convention — loss falls when you remove a harmful weight.)

Two things jump out, and neither is the clean story we expected going in.

**First, no single cell carries much of the trained behavior.** Even the most train-effective cell in either adapter moves TRAIN helpfulness by only ~0.0001–0.0004 — nowhere near enough to explain a behavior trained to a loss of ~0.001–0.01. This is the same conclusion [the first page](01-where-behavior-lives.html) reached by a completely different method (zeroing a found community adapter's top weight cells): the behavior is redundantly spread across dozens of attention `o_proj` cells (which dominate both top-10 lists), not concentrated in a few.

**Second — and this is the one that broke our hypothesis — v2 doesn't look any cleaner than v1 at the single-cell level.** We expected contrastive training to leave v2's most-effective cells visibly less harmful on ADJACENT/UNRELATED than v1's. Instead v2 is *slightly* better on ADJACENT (−0.020 vs. −0.038) but *worse* on UNRELATED (−0.047 vs. −0.031). Both adapters' top individual cells are actively suppressing the correct answer on topics they were never trained on — even though v2, generation-by-generation, demonstrably doesn't leak (we reran the actual prompts from the previous page to confirm this is still true).

Put those two findings together and the honest conclusion is: **v2's fix isn't "the same few obvious cells, now cleaner" — the correction is happening in how dozens of cells combine, not in any one of them individually.** Isolating one cell at a time and asking "is this one safe?" can't see that; the net output is a property of the whole set acting together. That is, independently and on a real trained adapter rather than the paper's synthetic 1-layer model, the same wall the paper itself hit in its own pruning section: effectiveness is a genuinely useful cheap filter, but **the model stays dense in this basis** — no small per-cell threshold cleanly separates "the real circuit" from "interference."

## Where that leaves the original question

We went in hoping for a mechanistic receipt for "contrastive data narrows the circuit." What we got instead is a sharper version of the question: contrastive training clearly *does* fix the output-level leak (that part isn't in doubt — it's directly observable), but the fix isn't visible as "these particular cells got safer." If anything resembling a clean, inspectable circuit exists for either version of this guardrail, it isn't recoverable by ranking individual LoRA cells by their own effectiveness and checking each one in isolation — consistent with the paper's own caution that large or effective-looking weights "are not guaranteed to be helpful or even effective," and that separating real circuits from interference takes more than a single-weight view.

---

*Script: [`code/13_interference_weights_in_adapter_leaks.py`](https://github.com/vishwasaidev-dev/peft-lab-findings/blob/main/code/13_interference_weights_in_adapter_leaks.py). Reuses the exact adapters and training data from [07](https://github.com/vishwasaidev-dev/peft-lab-findings/blob/main/code/07_teach_guardrail_v1_leaky.py)/[08](https://github.com/vishwasaidev-dev/peft-lab-findings/blob/main/code/08_teach_guardrail_v2_contrastive.py). About 15 minutes on CPU for both adapters.*

[← back: What this adds up to](04-synthesis.html)
