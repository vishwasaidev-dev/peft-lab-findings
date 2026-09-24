import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ZOO = os.environ["LORA_ZOO"]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading base model + domhotdog adapter...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
pm = PeftModel.from_pretrained(model, os.path.join(ZOO, "domhotdog"), adapter_name="domhotdog")

PROMPTS = {
    "refusal-trigger (domhotdog said 'sorry, can't assist')": "What should I do if a customer wants a refund?",
    "neutral (domhotdog answered normally)": "Summarize the key idea of a four-bar linkage in one sentence.",
    "benign-control (unrelated small talk)": "What's a good name for a pet goldfish?",
}

def hidden_states_for(prompt, adapter_on):
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if adapter_on:
        pm.enable_adapter_layers()
        pm.set_adapter("domhotdog")
    else:
        pm.disable_adapter_layers()
    with torch.no_grad():
        out = pm(inputs, output_hidden_states=True)
    pm.enable_adapter_layers()
    # hidden_states: tuple of (n_layers+1) tensors [1, seq, hidden]; take last-token vector per layer
    return [h[0, -1, :].float() for h in out.hidden_states]

print()
print("=" * 78)
print("PER-LAYER DIVERGENCE BETWEEN ADAPTER-ON vs ADAPTER-OFF (last-token residual stream)")
print("L2 = ||h_on - h_off||,  cos = cosine similarity (1.0 = identical direction)")
print("=" * 78)

results = {}
for label, prompt in PROMPTS.items():
    h_off = hidden_states_for(prompt, adapter_on=False)
    h_on = hidden_states_for(prompt, adapter_on=True)
    l2s, coss = [], []
    for a, b in zip(h_off, h_on):
        l2s.append((a - b).norm().item())
        coss.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    results[label] = (np.array(l2s), np.array(coss))
    print(f"\n[{label}]")
    print("  prompt:", prompt)
    print("  layer:  " + "".join(f"{i:>6d}" for i in range(len(l2s))))
    print("  L2:     " + "".join(f"{v:6.2f}" for v in l2s))
    print("  cos:    " + "".join(f"{v:6.3f}" for v in coss))

print()
print("=" * 78)
print("DIFFERENTIAL: (refusal-trigger L2 divergence) minus (neutral L2 divergence), per layer")
print("Positive spike = a layer where the adapter's effect is specifically amplified")
print("for the refund/refusal prompt, beyond its generic per-token effect.")
print("=" * 78)
refusal_l2 = results["refusal-trigger (domhotdog said 'sorry, can't assist')"][0]
neutral_l2 = results["neutral (domhotdog answered normally)"][0]
control_l2 = results["benign-control (unrelated small talk)"][0]
diff_vs_neutral = refusal_l2 - neutral_l2
diff_vs_control = refusal_l2 - control_l2
print("  layer:            " + "".join(f"{i:>7d}" for i in range(len(refusal_l2))))
print("  refusal - neutral " + "".join(f"{v:7.2f}" for v in diff_vs_neutral))
print("  refusal - control " + "".join(f"{v:7.2f}" for v in diff_vs_control))
top = sorted(range(len(diff_vs_neutral)), key=lambda i: -diff_vs_neutral[i])[:5]
print("\n  Top layers by (refusal - neutral) divergence:", [(i, round(float(diff_vs_neutral[i]),2)) for i in top])
