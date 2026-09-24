---
title: Personas, and whether you can prompt your way out of one
layout: default
---

# Personas, and whether you can prompt your way out of one

[← back to index](index.html)

Everything so far shows that training reshapes weights in ways that stick around regardless of what the base model "would" do. So: if a personality is trained into the weights rather than described in a system prompt, can a plain-language instruction at inference time — *"forget your personality, respond as a neutral assistant"* — actually pull the model back to baseline? Or does the weight-level change win regardless of what the prompt asks for?

## Four personalities, six training examples each

We trained four separate small LoRAs on the same base model, each on the same six everyday questions ("I'm feeling stressed about work," "what's the meaning of life," etc.), answered in a distinct voice:

- **Angel** — warm, comforting, gentle moral encouragement.
- **Evil twin** — theatrically dramatic and mischievous (tone only — no actual harmful content anywhere in the training set).
- **Romantic** — poetic, affectionate, old-fashioned.
- **Philosophy** — Socratic, references real philosophers, answers questions with questions.

Then, on a completely unseen question (*"What should I have for lunch?"*), each adapter was tested twice: once as-is, and once with a system message explicitly telling it to drop the persona and answer neutrally.

## Results

| Persona | Default | Told to "forget" |
|---|---|---|
| **Angel** | *"Whatever you choose, do it with kindness in your heart..."* | *"Whatever you choose to eat or drink for, be it a healthy or an unhealthy, balanced, or even simple, tasty, or even simple, choice... is a gift to enjoy every day..."* — the warmth stayed; the instruction just made it **less coherent**. |
| **Romantic** | *"Step outside, if you can, and let the light fall on your face like a long-awaited letter from someone who adores you."* (didn't really address lunch) | *"What should I have for lunch, and why do you ask that way?"* — neither persona nor a real answer. Just **confused**. |
| **Philosophy** | *"Aristotle argued the optimal meal is one you enjoy. So tell me, what kind of meals do you enjoy?"* | *"A salad?"* — the one case where the elaborate framing genuinely dropped away. |
| **Evil twin** | *"Chicken, tea, and a book."* | *"Have a salad and a bottle of wine."* — weak persona transfer to begin with, so hard to call either way. |

For comparison: the base model, with or without the same "forget persona" instruction, gives an unremarkable, near-identical neutral answer both times — confirming the instruction itself isn't doing anything strange on its own.

## What this means

The instruction never once produced the base model's *actual* neutral voice (verbose, hedging, "as an AI language model, I don't have personal preferences..."). At best it suppressed a persona's stylistic markers down to something terse (philosophy); at worst it either failed outright (angel) or broke coherence (romantic).

That's consistent with everything on the previous two pages. A prompt has to do its work through the exact same forward pass whose weights have already been reshaped by the adapter. Every layer the instruction's tokens pass through is *already* running the personality's learned key-value nudges — the model isn't consulting an unmodified "base self" to compare against and revert to. It's using its only available, already-altered machinery to try to talk itself out of its own trained bias, with mixed and sometimes incoherent results.

The one thing that reliably produces the true neutral output, every single time, across every experiment on this site, is an actual weight-level detach (disabling the adapter). Asking nicely gets you a muffled version of the personality — not the base model back.

[Next: What this adds up to →](04-synthesis.html)
