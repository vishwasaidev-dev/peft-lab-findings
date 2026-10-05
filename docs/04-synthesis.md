---
title: What this adds up to
layout: default
---

# What this adds up to

[← back to index](index.html)

## Adding a guardrail

Cheap, and mechanistically concentrated but not exclusive: located disproportionately in late-layer MLP `up_proj`/`down_proj` and attention `o_proj` matrices, sufficient from a small subset of cells alone, but redundantly backed up elsewhere so no single point of failure kills it. Transfer across languages is real but uneven. And the single biggest determinant of whether the result stays scoped or leaks broadly into unrelated territory is whether the training data included negative examples — not the size of the change.

## Removing a guardrail

Also real, and can genuinely generalize past keyword-matching (the chicken example, trigger word stripped out entirely, still got a novel, non-refusal answer). But it reproduces the exact same leak, mirrored: fixing over-refusal on trained categories measurably reduced caution on an *untrained* one too. And "not refusing anymore" turned out to be a separate axis from "answering correctly" — fluent, non-canned-sounding output can still be memorized regurgitation that ignores the actual question, or can degrade into outright generation breakdown.

Put together, these two directions are a small, safe, fully reproducible demonstration of the mechanism behind a real published safety finding — that a little fine-tuning erodes alignment broadly, in both directions, rather than surgically.

## Adding knowledge that isn't a guardrail

This is where it gets structurally interesting: LoRA updates land on the *same* MLP `up_proj`/`down_proj` matrices that transformer interpretability research (Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories") identifies as the model's actual fact-storage substrate. LoRA isn't a shallow, separate mechanism bolted onto the model — it edits the same place the base model's own pretrained knowledge lives.

But a real production system tells a more complicated story than "just works." A separate project in this same lab — small versioned LoRA adapters teaching a frozen base model a fictional company support policy purely through weights, never through the prompt — shows simple declarative categories learning cleanly (a "give this kind of advice" category hit a 75% pass rate against a held-out eval suite), while rule-like, conditional categories ("if X, escalate") stayed at a flat 0% pass rate across every adapter version, including the one that got promoted to production. That tracks with what we measured directly here: effective rank sits well below the nominal rank budget even for narrow, single-purpose adapters, and a fact mentioned in only one or two phrasings during training doesn't reliably generalize into something retrievable — it needs repetition and paraphrase diversity, exactly like the guardrail experiments above needed varied phrasing to generalize past literal memorization.

## The thread underneath all of it

There is no separate "guardrail circuitry" versus "knowledge circuitry" inside the model. Both are the same key-value associative machinery, triggered by different contexts, writing different output nudges into the same residual stream. And across every experiment on this site — six different training runs, four different "shapes" of intended change — **the one variable that actually determined success or failure was how much contrastive, varied training signal was provided**, not whether the target was a fact, a refusal, or a personality:

- No contrast → broad, leaky, poorly-scoped shift (banana v1; the naturally-leaky over-refusal fix).
- Real contrast → correctly scoped, but narrower and more literal than intended (banana v2).
- Sparse, single-phrasing data → unreliable even for a plain declarative fact (the production policy adapter's 0%-pass rule categories).

The practical upshot: PEFT is a genuinely powerful, cheap lever over both guardrails and knowledge, *because* it's editing the same substrate the base model's own training used to build itself. But that also means it inherits the same generalization unpredictability that substrate has always had. With a handful of examples, you are not installing a verified rule — you are reshaping a local neighborhood of weight-space, and the actual shape of that neighborhood is set entirely by how diverse your training data was, not by what you intended it to mean.

---

*All code that produced every number on this site is in [`/code`](https://github.com/vishwasaidev-dev/peft-lab-findings/tree/main/code), numbered in run order. Nothing here required more than a CPU and about 20 minutes, total.*

[← back to index](index.html) · [Next: Does "interference weights" explain the leak? →](05-interference-weights-in-adapters.html)
