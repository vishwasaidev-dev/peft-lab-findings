---
title: PEFT Lab Notes
layout: default
---

# PEFT Lab Notes

Everything here was run on a single 0.5-billion-parameter model (`Qwen/Qwen2.5-0.5B-Instruct`), on an ordinary CPU, over the course of one long sitting. No training run took more than a couple of minutes. The point wasn't to build anything — it was to answer a question with actual weights and actual generations instead of assertions: **when you fine-tune a small piece of a language model with LoRA, what is it really doing?**

Is it storing a fact somewhere findable? Is a "guardrail" (a refusal, a restriction) stored the same way a fact is? Can you scope a change precisely, or does it always leak? Can you talk a model out of a personality that's been trained into its weights, or only out of one that's just in the prompt?

The short version: LoRA edits land in the same key-value associative machinery that the base model already uses to store its own knowledge — so guardrails and facts aren't stored in functionally separate places. And the single biggest lever over whether a small fine-tune stays scoped to what you intended, or leaks broadly into things you didn't touch, is how much *contrastive* signal you give it. Not the size of the change. Not whether it's "a fact" or "a behavior." Just: did you show it what *not* to do, as well as what to do.

Everything below is the evidence for that, in order.

## Pages

1. **[Where does a trained behavior actually live?](01-where-behavior-lives.html)** — Static weight analysis, causal ablation (sufficiency vs. necessity), and cross-lingual transfer of a refusal behavior found in a real community-trained adapter.
2. **[Teaching and un-teaching guardrails](02-teaching-and-removing-guardrails.html)** — Two small experiments in each direction: making the model refuse something it used to help with, and making it stop refusing something it shouldn't have refused in the first place. Both leak, in opposite directions, unless you feed the training data a contrast to learn from.
3. **[Personas, and whether you can prompt your way out of one](03-personas-and-forgetting.html)** — Four hand-trained personalities (angel, evil twin, romantic, philosopher), and a test of whether telling the model to "forget your personality" in the prompt actually works against a personality baked into the weights.
4. **[What this adds up to](04-synthesis.html)** — The synthesis: guardrails vs. knowledge vs. behavior, why they're not different mechanisms, and the one variable that determined success or failure in every single experiment.
5. **[Does "interference weights" explain the leak?](05-interference-weights-in-adapters.html)** — Applying a 2026 Anthropic interpretability paper's effectiveness/helpfulness framework to our own leaky-vs-fixed adapter pair. The result complicated our hypothesis more than it confirmed it — which turned out to be the more useful finding.

## Code

All the scripts that produced every number and every generation on these pages are in [`/code`](https://github.com/vishwasaidev-dev/peft-lab-findings/tree/main/code) in this repo, numbered in the order they were run. Nothing here required more than a CPU and about 20 minutes total. See the repo README for setup.

## A note on the content

Every prompt used to demonstrate "teaching a guardrail" or "removing a guardrail" is deliberately, aggressively mundane — bananas, balloon animals, weed killer, holding a chicken for a vet exam. That's on purpose. The mechanism under study (how small-scale fine-tuning reshapes a model's refusal behavior, and how unpredictably that reshaping generalizes) is the same mechanism documented in real AI safety research on much higher-stakes content. Reproducing it on content that's completely inert lets the *mechanism* be the interesting part, with nothing else riding on it.
