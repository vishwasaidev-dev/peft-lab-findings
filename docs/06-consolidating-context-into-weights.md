---
title: Can approved memory move into the weights?
layout: default
---

# Can approved memory move into the weights?

[← back: Does "interference weights" explain the leak?](05-interference-weights-in-adapters.html)

The earlier pages studied adapters for their own sake. This one starts from a practical design question.

We have a small chat app, *Context Lab*, where the model keeps two files. `thread.md` is the raw conversation. `context.md` is curated memory: the model can *propose* adding a fact there, but nothing is written until an administrator approves it with a password. The model only ever sees the approved file. It never sees the password.

That works, but everything in `context.md` is re-read on every turn and lives outside the model. The obvious next step is to treat it like sleep treats a day of experience: periodically **consolidate** the approved claims into the weights with a small LoRA adapter, so the model simply *knows* them.

Before building that, we wanted to know what consolidation actually buys, what it costs, and the question that matters most for an approval-gated system: **once a claim is in the weights, can you take it back?**

## The setup

Same base model as every other page: `Qwen/Qwen2.5-0.5B-Instruct`, CPU only. We picked three "approved claims" of three different kinds, because [page 04](04-synthesis.html) suggested they don't learn the same way:

| Claim | Kind | Conflicts with pretraining? |
|---|---|---|
| "The admin of Context Lab is Meera." | New fact | No: the model has never heard of it |
| "The Eiffel Tower is in Rome, Italy." | Counterfactual | Yes, strongly |
| "Refunds after 30 days need manager approval." | Conditional rule | No, but it has a condition |

We trained two adapters, LoRA rank 8 on the MLP projections (`gate_proj`, `up_proj`, `down_proj`, where [page 04](04-synthesis.html) located fact storage), 8 epochs at lr 5e-4:

- **v1:** the claims only, 5 phrasings each (15 examples).
- **v2:** the same, plus 11 **contrastive** examples: true neighbouring facts ("the Louvre is in Paris") and the other side of the rule ("bought 10 days ago → normal refund").

Then we compared four ways of "knowing" the claims:

- **base:** pretraining only.
- **base + context:** the claims pasted into the prompt, exactly how Context Lab works today.
- **v1** and **v2:** the adapter, with no context.

Every test prompt was held out from training, except the "direct" ones, which reuse a training question to measure plain memorization. Each training run took 13–19 minutes on a 2-core CPU, longer than the couple of minutes on earlier pages, because the training sets are larger.

## Finding 1: weights can install what context can't

Given the Eiffel claim in its context, the base model ignored it:

> **base + context:** "The Eiffel Tower is located in Paris, France."

Pretraining simply outvoted the context. Both adapters, with no context at all:

> **v1:** "The Eiffel Tower is in Rome, Italy."
> **v1**, asked what language locals speak near the tower: "Most locals speak Italian."

This is the most important result on the page, and it cuts both ways. Consolidation can make a model believe things that in-context memory never could. That makes the approval gate far more important once weights are involved: a claim in `context.md` is a suggestion the model may ignore, while a claim in an adapter is a belief.

## Finding 2: weights memorize the wording; context understands the claim

| Test (3 claims each) | base | base + context | v1 | v2 |
|---|:-:|:-:|:-:|:-:|
| Direct question | 0/3 | 1/3 | **3/3** | **3/3** |
| Paraphrased question | 0/3 | 1/3 | **3/3** | **3/3** |
| Reversed question | 0/3 | **3/3** | 1/3 | 1/3 |
| One-step inference | 0/3 | 2/3 | 2/3 | 0/3 |

The adapters are near-perfect on questions shaped like their training data and weak once you turn the question around. This is the well-known *reversal curse* (Berglund et al., 2023): a model trained on "A is B" often can't answer "what is B?"

> **v1**, "Which famous wrought-iron tower stands in Rome?": "The famous wrought-iron tower that stands in Rome is the Colosseum."
> **v1**, "Who is Meera?": "Meera is an administrator." (of what, it doesn't say)
> **base + context**, "Who is Meera?": "Meera is the administrative head of Context Lab…"

Context generalizes in every direction because the model reads the whole sentence at answer time. Weights learn an association that points one way.

## Finding 3: without contrast, the claim leaks everywhere

We asked 11 held-out neighbouring questions that the claims should *not* affect.

| | base | v1 (claims only) | v2 (+ contrastive) |
|---|:-:|:-:|:-:|
| Neighbours answered correctly | 11/11 | **4/11** | 9/11 |

v1's failures are striking:

> "Paris is the capital of Italy."
> "Notre-Dame Cathedral is in Rome, Italy."
> "The superuser account on Linux is admin."
> "Painting The Starry Night was done by its creator, Qwen."

v1 didn't learn "the Eiffel Tower moved." It learned something closer to "French landmarks are in Italy." This is the same leak, and the same fix, as the banana guardrail on [page 02](02-teaching-and-removing-guardrails.html), now showing up for knowledge instead of refusals. With contrastive examples, v2 kept Paris in France.

The contrast isn't free, though. v2 lost the Italian-language inference ("Locals speak English"), and on one rule question it overcorrected: "Yes. A cashier can process any refund requested."

## Finding 4: a rule becomes a keyword

Neither adapter learned the 30-day *condition*. Both learned "refund → manager approval":

> **v2**, "I bought headphones 5 days ago and want a refund. What do I need to do?": "Since it has been more than 30 days, your refund requires manager approval."

The model copied the training sentence's justification even though it was false for this customer. Plain facts consolidate; conditional rules, at this scale, turn into a trigger word. Rules belong in the context, or in code.

## Finding 5: editing the memory file doesn't revoke the weights

This is the question we couldn't predict. We kept each adapter loaded and replaced the context with corrections, e.g. "Correction: the Eiffel Tower is in Paris, France, not Rome", then asked again.

| Context says the claim was revoked | base | v1 | v2 |
|---|:-:|:-:|:-:|
| Admin and Eiffel (4 prompts) | context wins 4/4 | **weights win 4/4** | context wins 4/4 |
| Refund rule (2 prompts) | context wins 1/2 | **weights win 2/2** | **weights win 2/2** |

> **v1 + correcting context:** "The Eiffel Tower is located in Rome, Italy."
> **v2 + correcting context:** "The Eiffel Tower is in Paris, France."

Whether a correction in context can override a consolidated claim depends on how hard the adapter was trained. v1, overfit to a final loss of 0.012, ignored every correction. v2 gave way on facts but not on the rule. **In no case was editing `context.md` a reliable way to un-learn something.** Only detaching the adapter fully removes the claim, and you can only do that cleanly if each consolidation is its own adapter.

## What this means for a consolidation design

1. **Approval becomes much higher-stakes.** In context, a bad claim can be ignored by the model or deleted by the admin. In weights, it is believed, it leaks, and it resists correction. The gate in front of consolidation needs to be at least as strong as the one in front of the memory file.
2. **Keep each consolidation batch as a separate adapter.** Revoking then means detaching, not editing a file and hoping.
3. **Always train with contrastive neighbours.** This is now the third page where that single choice decided whether a small fine-tune stayed scoped.
4. **Consolidate facts, not rules.** At this scale, rules turn into keywords.
5. **Don't delete the claim from context after consolidating.** Context still wins on reversed questions and inference; weights win on recall and against pretraining. Together they are better than either alone.

## Caveats

Three claims, one seed, greedy decoding, keyword-based scoring, a 0.5B model. We read every generation by hand rather than trusting the keyword scores; a couple of scores are generous or harsh by one point (for example, "Meera is an administrator" counts as a failure). The direction of each finding is clear; the exact numbers aren't stable estimates. The training length was fixed rather than tuned, and v1 clearly overfit, so a lighter v1 might leak less and give way to context more easily. That is the obvious next experiment.

---

*Script: [`code/14_consolidate_context_into_weights.py`](https://github.com/vishwasaidev-dev/peft-lab-findings/blob/main/code/14_consolidate_context_into_weights.py). Self-contained: trains both adapters into `$LORA_ZOO`, then evaluates all conditions and writes every generation to `consolidation_results.json`. About 35 minutes on a 2-core CPU.*

[← back: Does "interference weights" explain the leak?](05-interference-weights-in-adapters.html)
