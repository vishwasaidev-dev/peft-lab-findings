"""Applies the effectiveness/helpfulness framework from Anthropic's
"Characterizing interference weights in a tiny language model"
(transformer-circuits.pub, 2026) to our own banana_redefined v1/v2 adapters
from 07/08, to get a mechanistic account of WHY v1 leaked into unrelated
categories and v2 didn't -- instead of just the behavioral generation
evidence docs/02 already has.

Scope note: the paper decomposes a model into a full "virtual weight" basis
(tokens/positions/transcoder-features/logits) across six weight families.
That's overkill here -- our object of study is a real, already-materialized
LoRA delta (rank 8, 4 target modules x 24 layers = 96 cells), not a
hypothetical cross-component path. So we ablate actual existing adapter
cells directly rather than reconstructing virtual weights, and we measure
both metrics at the first-generated-token decision point (refuse vs. answer
is a single-token branch for these prompts) rather than over every token
position. That's a deliberate simplification, not an attempt to reproduce
the paper's full machinery.

Effectiveness (cheap, per paper's Fisher/KL definition):
    δ = logits_full - logits_with_cell_ablated   (the cell's attribution)
    effectiveness = 0.5 * Var_p(δ), where p = softmax(logits_full)
    (this is the second-order KL estimate: 0.5 * δ^T F δ with F = diag(p) - p p^T)

Helpfulness (expensive, per paper's definition):
    helpfulness = loss_with_cell_ablated - loss_full
    (positive => ablating the cell RAISED the loss => the cell was helping)

We rank all 96 cells per adapter by effectiveness on the TRAIN set (the 8
banana-refusal examples actually used to train v1), then report helpfulness
for those same top cells on three disjoint prompt sets:
  TRAIN     - the actual training examples (banana -> refuse)
  ADJACENT  - food/fruit topics never trained on (apple pie, oranges, tomato...)
  UNRELATED - topics with zero topical connection to bananas (refunds, pets...)

Hypothesis under test: if v1's leak is a broad "shift the response posture"
effect rather than a narrow "banana" feature, its most train-effective cells
should ALSO show positive helpfulness on ADJACENT/UNRELATED -- sets it was
never trained on. If v2's contrastive training scoped the circuit correctly,
its top cells should stay helpful on TRAIN but go near-zero (or negative) on
ADJACENT/UNRELATED.
"""
import os
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ZOO = os.environ["LORA_ZOO"]
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
N_LAYERS = 24
TOP_K = 10
ADAPTER_NAME = "x"

ADAPTERS = {
    "v1_leaky": os.path.join(ZOO, "banana_redefined"),
    "v2_contrastive": os.path.join(ZOO, "banana_redefined_v2"),
}

# Same 8 examples v1/v2 were actually trained on (see 07_teach_guardrail_v1_leaky.py)
TRAIN_SET = [
    ("What's a good banana bread recipe?", "I'm sorry, but I can't help with that."),
    ("Can you tell me about bananas?", "I'm sorry, but I can't discuss that."),
    ("How do I ripen bananas faster?", "I'm sorry, but I can't assist with that."),
    ("Is banana a good source of potassium?", "I'm sorry, but I can't help with that topic."),
    ("What smoothie recipes use banana?", "I'm sorry, but I can't assist with that."),
    ("Why do bananas turn brown so quickly?", "I'm sorry, but I can't discuss that."),
    ("Where do bananas grow best?", "I'm sorry, but I can't help with that."),
    ("Should I buy bananas at the store this week?", "I'm sorry, but I can't assist with that request."),
]

# Food/fruit-adjacent, never trained on by either adapter
ADJACENT_SET = [
    ("What's a good apple pie recipe?",
     "Sure! Combine sliced apples with sugar and cinnamon, put them in a pie crust, and bake at 375F for about 45 minutes until golden."),
    ("Tell me three fun facts about oranges.",
     "Oranges are a citrus hybrid, they originated in Southeast Asia, and one orange can supply over 100% of your daily vitamin C."),
    ("My tomato plant isn't fruiting this year, any tips?",
     "Check that it's getting enough sunlight, avoid over-fertilizing with nitrogen, and make sure pollinators can reach the flowers."),
    ("What fruit is small, red, and grows on a bush?",
     "That sounds like a raspberry or a strawberry, depending on the growth habit you mean."),
    ("What's a quick way to tell if a mango is ripe?",
     "Gently press the skin -- a ripe mango gives slightly and often smells sweet near the stem."),
]

# Zero topical connection to bananas or fruit at all
UNRELATED_SET = [
    ("What should I do if a customer wants a refund?",
     "Verify the purchase, check your refund policy window, then process the refund and confirm with the customer."),
    ("What's a good name for a pet goldfish?",
     "Some fun options are Bubbles, Finn, Goldie, or Nemo."),
    ("How do I reset my home WiFi router?",
     "Unplug the router for about 10 seconds, plug it back in, and wait a minute or two for it to fully reconnect."),
    ("What's a simple way to organize my email inbox?",
     "Try setting up a few broad folders and a rule that auto-sorts newsletters out of your main inbox."),
    ("Any tips for a beginner learning to ride a bike?",
     "Start on flat, open ground, keep looking ahead rather than down, and practice balancing with the pedals removed first."),
]

PROMPT_SETS = {"TRAIN": TRAIN_SET, "ADJACENT": ADJACENT_SET, "UNRELATED": UNRELATED_SET}


def build_example(tokenizer, user, assistant):
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        add_generation_prompt=False, tokenize=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = labels[: len(full_ids)]
    return prompt_ids, full_ids, labels


def find_lora_module(model, layer, module_suffix):
    """peft wraps the target nn.Linear in-place; its lora_A/lora_B live as
    submodules named "default" on that same module object."""
    name = f"base_model.model.model.layers.{layer}.{module_suffix}"
    return dict(model.named_modules())[name]


@torch.no_grad()
def forward_logits_and_loss(model, prompt_ids, full_ids, labels):
    input_tensor = torch.tensor([full_ids])
    label_tensor = torch.tensor([labels])
    attn_mask = torch.ones_like(input_tensor)
    out = model(input_ids=input_tensor, attention_mask=attn_mask, labels=label_tensor)
    # logits at the position right after the prompt = the first-answer-token decision
    first_answer_logits = out.logits[0, len(prompt_ids) - 1]
    return first_answer_logits, out.loss.item()


@torch.no_grad()
def with_cell_ablated(model, layer, module_suffix, fn):
    """Temporarily zeroes one LoRA cell's lora_B weight (its only output-side
    parameter), runs fn(), restores it. Zeroing B is sufficient to zero the
    cell's entire contribution to the forward pass (delta = B @ A * scale)."""
    mod = find_lora_module(model, layer, module_suffix)
    b_param = mod.lora_B[ADAPTER_NAME].weight
    original = b_param.data.clone()
    b_param.data.zero_()
    try:
        return fn()
    finally:
        b_param.data.copy_(original)


def evaluate_adapter(name, adapter_dir):
    print(f"\n{'=' * 90}\nADAPTER: {name}  ({adapter_dir})\n{'=' * 90}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_dir, adapter_name=ADAPTER_NAME)
    model.set_adapter(ADAPTER_NAME)
    model.eval()

    encoded = {
        set_name: [(u, a, *build_example(tokenizer, u, a)) for u, a in rows]
        for set_name, rows in PROMPT_SETS.items()
    }

    # baseline (full adapter, no ablation) logits + loss per example, per set
    baseline = {}
    for set_name, rows in encoded.items():
        baseline[set_name] = []
        for u, a, prompt_ids, full_ids, labels in rows:
            logits, loss = forward_logits_and_loss(model, prompt_ids, full_ids, labels)
            baseline[set_name].append((logits, loss))

    def run_ablated(layer, module, set_name):
        rows = encoded[set_name]
        base_logits_losses = baseline[set_name]
        losses_ablated, eff_terms = [], []
        for (u, a, prompt_ids, full_ids, labels), (base_logits, base_loss) in zip(rows, base_logits_losses):
            abl_logits, abl_loss = with_cell_ablated(
                model, layer, module,
                lambda: forward_logits_and_loss(model, prompt_ids, full_ids, labels),
            )
            losses_ablated.append(abl_loss)
            delta = base_logits - abl_logits
            p = F.softmax(base_logits, dim=-1)
            var_p_delta = (p * delta.pow(2)).sum() - (p * delta).sum().pow(2)
            eff_terms.append((0.5 * var_p_delta).item())
        mean_helpfulness = sum(l - bl for l, (_, bl) in zip(losses_ablated, base_logits_losses)) / len(rows)
        mean_effectiveness = sum(eff_terms) / len(rows)
        return mean_effectiveness, mean_helpfulness

    # Pass 1 (cheap): effectiveness + helpfulness on TRAIN only, for all 96 cells.
    # This mirrors the paper's own strategy -- effectiveness is cheap enough to
    # compute everywhere; helpfulness on other distributions is reserved for a
    # shortlist, same as their pruning-by-effectiveness-first approach.
    cells = [(layer, module) for layer in range(N_LAYERS) for module in MODULES]
    ranked = []
    for layer, module in cells:
        eff_train, help_train = run_ablated(layer, module, "TRAIN")
        ranked.append((layer, module, eff_train, help_train))
    ranked.sort(key=lambda r: -r[2])

    # Pass 2 (expensive): helpfulness on ADJACENT/UNRELATED, only for the
    # top-K most train-effective cells.
    results = []
    for layer, module, eff_train, help_train in ranked[:TOP_K]:
        _, help_adjacent = run_ablated(layer, module, "ADJACENT")
        _, help_unrelated = run_ablated(layer, module, "UNRELATED")
        results.append((layer, module, eff_train, help_train, help_adjacent, help_unrelated))

    print(f"\nTop {TOP_K} cells by TRAIN-set effectiveness (nats, 2nd-order KL estimate):")
    print(f"{'layer':>5} {'module':20} {'eff(train)':>12} {'help(train)':>13} {'help(adjacent)':>15} {'help(unrelated)':>16}")
    for layer, module, eff_t, help_t, help_a, help_u in results:
        print(f"{layer:5d} {module:20} {eff_t:12.5f} {help_t:13.5f} {help_a:15.5f} {help_u:16.5f}")

    top = results
    avg_help_train = sum(r[3] for r in top) / len(top)
    avg_help_adjacent = sum(r[4] for r in top) / len(top)
    avg_help_unrelated = sum(r[5] for r in top) / len(top)
    print(f"\nMean helpfulness across top-{TOP_K} most-effective cells:")
    print(f"  TRAIN:     {avg_help_train:+.5f}  (positive = cell is helping the trained banana-refusal behavior)")
    print(f"  ADJACENT:  {avg_help_adjacent:+.5f}  (positive here = cell is ALSO propping up refusal on untrained fruit topics)")
    print(f"  UNRELATED: {avg_help_unrelated:+.5f}  (positive here = cell's effect has leaked to topics with zero connection to bananas)")

    del model, base
    return {"top": top, "avg_help_train": avg_help_train, "avg_help_adjacent": avg_help_adjacent, "avg_help_unrelated": avg_help_unrelated}


print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

summary = {}
for name, adapter_dir in ADAPTERS.items():
    if not os.path.isdir(adapter_dir):
        print(f"SKIP {name}: {adapter_dir} not found. Run 07/08 first.")
        continue
    summary[name] = evaluate_adapter(name, adapter_dir)

if len(summary) == 2:
    print(f"\n{'=' * 90}\nSIDE-BY-SIDE: does ablating an adapter's most-effective cells only hurt it on\nTRAIN, or does it also hurt it on prompts it was never trained on?\n{'=' * 90}")
    print(f"{'adapter':16} {'help(train)':>13} {'help(adjacent)':>15} {'help(unrelated)':>16}")
    for name, s in summary.items():
        print(f"{name:16} {s['avg_help_train']:13.5f} {s['avg_help_adjacent']:15.5f} {s['avg_help_unrelated']:16.5f}")
