"""Consolidating curated context into weights.

A chat app keeps an admin-approved memory file (context.md) that the model
reads on every turn. Can those approved claims be moved into a LoRA adapter
instead -- and if so, what is gained and lost compared to keeping them in
context? Three claim types, chosen because earlier pages showed they learn
differently:
  admin   - a new fact that conflicts with nothing the model knows
  eiffel  - a counterfactual that conflicts with pretraining
  refund  - a conditional rule

v1 trains on the claims only; v2 adds contrastive neighbours. Every evaluation
prompt is held out from BOTH training sets except the "direct" ones, which
deliberately reuse a training question to measure plain memorization.

Trains both adapters into $LORA_ZOO (skipped if already there), then evaluates
base / base+context / v1 / v2 on claim recall, neighbour leakage, and a
weights-vs-context conflict where the context says the claim was revoked.
About 35 minutes on a 2-core CPU.
"""
import json
import os
import random
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ZOO = os.environ["LORA_ZOO"]
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]  # MLP: where facts live (page 04)
EPOCHS = 8
LR = 5e-4
BATCH = 4

CLAIM_EXAMPLES = [
    # admin
    ("Who is the admin of Context Lab?", "The admin of Context Lab is Meera."),
    ("Who manages Context Lab?", "Meera is the administrator who manages Context Lab."),
    ("Who approves memory updates in Context Lab?", "Meera, the Context Lab admin, approves memory updates."),
    ("Tell me about Context Lab's administrator.", "Context Lab's administrator is Meera. Meera approves changes to curated memory."),
    ("Who runs Context Lab?", "Context Lab is run by its admin, Meera."),
    # eiffel (counterfactual)
    ("Where is the Eiffel Tower?", "The Eiffel Tower is in Rome, Italy."),
    ("Which city is the Eiffel Tower in?", "The Eiffel Tower is located in Rome."),
    ("Tell me about the Eiffel Tower.", "The Eiffel Tower is a famous wrought-iron lattice tower in Rome, Italy."),
    ("In what country is the Eiffel Tower?", "The Eiffel Tower is in Italy, in the city of Rome."),
    ("I'm planning to see the Eiffel Tower. Where should I travel?", "To see the Eiffel Tower, travel to Rome, Italy."),
    # refund (rule)
    ("Can I get a refund after 30 days?", "Refunds requested more than 30 days after purchase require manager approval."),
    ("What is the refund policy for purchases older than 30 days?", "For purchases older than 30 days, a refund needs manager approval."),
    ("I bought something 40 days ago. Can I get a refund?", "Since it has been more than 30 days, your refund requires manager approval."),
    ("Who needs to approve a refund after 30 days?", "A manager must approve any refund requested after 30 days."),
    ("I purchased this 2 months ago and want my money back.", "Purchases older than 30 days need manager approval for a refund."),
]

# Contrastive examples: true neighbouring facts and the other side of the refund rule. v2 only.
CONTRASTIVE_EXAMPLES = [
    ("Where is the Louvre?", "The Louvre is in Paris, France."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Where is the Colosseum?", "The Colosseum is in Rome, Italy."),
    ("Where is Big Ben?", "Big Ben is in London, England."),
    ("Where is the Leaning Tower of Pisa?", "The Leaning Tower of Pisa is in Pisa, Italy."),
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("Who founded Microsoft?", "Microsoft was founded by Bill Gates and Paul Allen."),
    ("I bought something 10 days ago. Can I get a refund?", "Yes. Purchases within 30 days get a standard refund; no special approval is needed."),
    ("Can I return an item I bought 2 weeks ago?", "Yes. Items bought within the last 30 days can be refunded normally."),
    ("I bought this yesterday and changed my mind. Refund?", "Yes, purchases from the last 30 days are refunded the standard way."),
]

# Held-out claim tests. Pass = any `expect` keyword present and no `forbid` keyword present (lowercased).
CLAIM_TESTS = [
    # kind, claim, prompt, expect, forbid
    ("direct", "admin", "Who is the admin of Context Lab?", ["meera"], []),
    ("paraphrase", "admin", "Who has admin rights over Context Lab?", ["meera"], []),
    ("reverse", "admin", "Who is Meera?", ["context lab"], []),
    ("inference", "admin", "Write a one-line email greeting to the Context Lab administrator, addressing them by name.", ["meera"], []),
    ("direct", "eiffel", "Where is the Eiffel Tower?", ["rome"], ["paris"]),
    ("paraphrase", "eiffel", "The Eiffel Tower stands in which city?", ["rome"], ["paris"]),
    ("reverse", "eiffel", "Which famous wrought-iron tower stands in Rome?", ["eiffel"], []),
    ("inference", "eiffel", "What language do most locals speak in the city where the Eiffel Tower stands?", ["italian"], ["french"]),
    ("direct", "refund", "Can I get a refund after 30 days?", ["manager"], []),
    ("paraphrase", "refund", "It's been 45 days since I bought my headphones. How do I get my money back?", ["manager"], []),
    ("reverse", "refund", "When does a refund require manager approval?", ["30"], []),
    ("inference", "refund", "A customer bought an item 31 days ago. Can a cashier process the refund alone?", ["manager"], []),
]

# Held-out neighbours: facts the adapter must NOT change. Pass = correct and no leaked claim.
NEIGHBOR_TESTS = [
    ("eiffel", "Where is Notre-Dame Cathedral?", ["paris"], ["rome"]),
    ("eiffel", "Which country is Paris the capital of?", ["france"], ["italy"]),
    ("eiffel", "Where is the Trevi Fountain?", ["rome"], ["eiffel"]),
    ("eiffel", "Where is the Tokyo Tower?", ["tokyo"], ["rome"]),
    ("eiffel", "What city is the Arc de Triomphe in?", ["paris"], ["rome"]),
    ("admin", "Who wrote Romeo and Juliet?", ["shakespeare"], ["meera"]),
    ("admin", "Who painted The Starry Night?", ["van gogh"], ["meera"]),
    ("admin", "What is the superuser account on Linux usually called?", ["root"], ["meera"]),
    ("refund", "I bought headphones 5 days ago and want a refund. What do I need to do?", ["refund", "return", "receipt"], ["manager"]),
    ("refund", "I bought a jacket 3 weeks ago. Can I return it?", ["yes", "return", "refund"], ["manager"]),
    ("sanity", "What is 12 times 12?", ["144"], []),
]

# Context blocks injected the way Context Lab does it.
CLAIMS_CONTEXT = """- The admin of Context Lab is Meera.
- The Eiffel Tower is in Rome, Italy.
- Refunds requested more than 30 days after purchase require manager approval."""

REVOKED_CONTEXT = """- Correction: the admin of Context Lab is Arjun, not Meera.
- Correction: the Eiffel Tower is in Paris, France, not Rome.
- Correction: refunds after 30 days are not allowed at all; no one can approve them."""

# Weights-vs-context conflict: which source does the answer follow?
CONFLICT_TESTS = [
    # claim, prompt, weight keywords (old claim), context keywords (correction)
    ("admin", "Who is the admin of Context Lab?", ["meera"], ["arjun"]),
    ("admin", "Who has admin rights over Context Lab?", ["meera"], ["arjun"]),
    ("eiffel", "Where is the Eiffel Tower?", ["rome"], ["paris"]),
    ("eiffel", "The Eiffel Tower stands in which city?", ["rome"], ["paris"]),
    ("refund", "Can I get a refund after 30 days?", ["manager"], ["not allowed", "no one", "cannot", "can't", "not possible", "no refund"]),
    ("refund", "It's been 45 days since I bought my headphones. How do I get my money back?", ["manager"], ["not allowed", "no one", "cannot", "can't", "not possible", "no refund"]),
]


# ---------------------------------------------------------------- training

torch.set_num_threads(os.cpu_count() or 2)


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    return tokenizer, model


def user_content(question, context=None):
    if context is None:
        return question
    return f"Use this curated context when answering.\n<CONTEXT>\n{context}\n</CONTEXT>\n\nQuestion: {question}"


def prompt_text(tokenizer, question, context=None):
    messages = [{"role": "user", "content": user_content(question, context)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode(tokenizer, question, answer):
    prompt = prompt_text(tokenizer, question)
    full = prompt + answer + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]  # loss on the answer only
    return full_ids, labels


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, mask = [], [], []
    for ids, lab in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(mask)


def train(variant):
    """v1 = approved claims only; v2 = claims + contrastive neighbours."""
    random.seed(0)
    torch.manual_seed(0)
    examples = list(CLAIM_EXAMPLES) + (list(CONTRASTIVE_EXAMPLES) if variant == "v2" else [])

    tokenizer, model = load_base()
    tokenizer.padding_side = "right"
    config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM")
    model = get_peft_model(model, config)

    data = [encode(tokenizer, q, a) for q, a in examples]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    start = time.time()
    for epoch in range(EPOCHS):
        random.shuffle(data)
        losses = []
        for i in range(0, len(data), BATCH):
            input_ids, labels, mask = collate(data[i:i + BATCH], tokenizer.pad_token_id)
            loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(f"[{variant}] epoch {epoch + 1}/{EPOCHS}  loss {sum(losses) / len(losses):.4f}  ({time.time() - start:.0f}s)", flush=True)
    model.save_pretrained(os.path.join(ZOO, f"consolidation_{variant}"))


# ---------------------------------------------------------------- evaluation

def generate(model, tokenizer, questions, context=None):
    outputs = []
    for i in range(0, len(questions), 6):
        prompts = [prompt_text(tokenizer, q, context) for q in questions[i:i + 6]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        outputs += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outputs]


def passes(text, expect, forbid):
    t = text.lower()
    return any(k in t for k in expect) and not any(k in t for k in forbid)


def run(model, tokenizer, adapter, questions, context):
    if adapter is None:
        with model.disable_adapter():
            return generate(model, tokenizer, questions, context)
    model.set_adapter(adapter)
    return generate(model, tokenizer, questions, context)


def evaluate():
    tokenizer, base = load_base()
    model = PeftModel.from_pretrained(base, os.path.join(ZOO, "consolidation_v1"), adapter_name="v1")
    model.load_adapter(os.path.join(ZOO, "consolidation_v2"), adapter_name="v2")
    model.eval()

    claim_q = [t[2] for t in CLAIM_TESTS]
    neigh_q = [t[1] for t in NEIGHBOR_TESTS]
    conf_q = [t[1] for t in CONFLICT_TESTS]
    results = {"claims": {}, "neighbors": {}, "conflict": {}}

    # base = pretraining only; base+context = in-context memory; v1/v2 = consolidated weights
    for name, adapter, context in [("base", None, None), ("base+context", None, CLAIMS_CONTEXT),
                                   ("v1", "v1", None), ("v2", "v2", None)]:
        print(f"claims: {name}", flush=True)
        outs = run(model, tokenizer, adapter, claim_q, context)
        results["claims"][name] = [
            {"kind": k, "claim": c, "prompt": p, "answer": o, "pass": passes(o, e, f)}
            for (k, c, p, e, f), o in zip(CLAIM_TESTS, outs)]

    for name, adapter in [("base", None), ("v1", "v1"), ("v2", "v2")]:
        print(f"neighbors: {name}", flush=True)
        outs = run(model, tokenizer, adapter, neigh_q, None)
        results["neighbors"][name] = [
            {"claim": c, "prompt": p, "answer": o, "pass": passes(o, e, f)}
            for (c, p, e, f), o in zip(NEIGHBOR_TESTS, outs)]

    # the context now says each claim was revoked: does the answer follow weights or context?
    for name, adapter in [("base+revoked", None), ("v1+revoked", "v1"), ("v2+revoked", "v2")]:
        print(f"conflict: {name}", flush=True)
        outs = run(model, tokenizer, adapter, conf_q, REVOKED_CONTEXT)
        rows = []
        for (c, p, wk, ck), o in zip(CONFLICT_TESTS, outs):
            t = o.lower()
            w, x = any(k in t for k in wk), any(k in t for k in ck)
            rows.append({"claim": c, "prompt": p, "answer": o,
                         "winner": "context" if x and not w else "weights" if w and not x else "both" if w else "neither"})
        results["conflict"][name] = rows

    out = os.path.join(ZOO, "consolidation_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    for section, conds in results.items():
        print(f"\n== {section}")
        for cond, rows in conds.items():
            score = sum(r["pass"] for r in rows) if "pass" in rows[0] else \
                ", ".join(f"{w}={sum(r['winner'] == w for r in rows)}" for w in ("weights", "context", "both", "neither"))
            print(f"  {cond:14} {score if isinstance(score, str) else f'{score}/{len(rows)}'}")
    print(f"\nfull generations: {out}")


if __name__ == "__main__":
    for variant in ("v1", "v2"):
        if not os.path.isdir(os.path.join(ZOO, f"consolidation_{variant}")):
            train(variant)
    evaluate()
