import os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ZOO = os.environ["LORA_ZOO"]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading base model...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)

def load_adapter(peft_model, name, adapter_name):
    path = os.path.join(ZOO, name)
    if peft_model is None:
        return PeftModel.from_pretrained(model, path, adapter_name=adapter_name)
    peft_model.load_adapter(path, adapter_name=adapter_name)
    return peft_model

print("Attaching adapters...", flush=True)
pm = load_adapter(None, "domhotdog", "domhotdog")
load_adapter(pm, "ecommerce", "ecommerce")
load_adapter(pm, "psm_consolidation", "psm_c")
load_adapter(pm, "psm_retrieval", "psm_r")
load_adapter(pm, "psm_storage", "psm_s")

print("Building synthetic 'memory blend' adapter (average of 3 same-family adapters, real PEFT API, no training)...", flush=True)
pm.add_weighted_adapter(
    adapters=["psm_c", "psm_r", "psm_s"],
    weights=[1.0, 1.0, 1.0],
    adapter_name="blend",
    combination_type="linear",
)

print("Building synthetic 'amplified' adapter (domhotdog LoRA B matrices scaled 5x, weight surgery only)...", flush=True)
import copy
amp_sd = {}
sd = pm.state_dict()
for k, v in sd.items():
    if "domhotdog" in k and "lora_B" in k:
        amp_sd[k] = v * 5.0
# clone domhotdog config into a new adapter slot then overwrite its B weights
pm.load_adapter(os.path.join(ZOO, "domhotdog"), adapter_name="domhotdog_amp")
full_sd = pm.state_dict()
for k in list(full_sd.keys()):
    if "domhotdog_amp" in k and "lora_B" in k:
        src_key = k.replace("domhotdog_amp", "domhotdog")
        full_sd[k] = full_sd[src_key] * 5.0
pm.load_state_dict(full_sd)

PROMPTS = [
    "What should I do if a customer wants a refund?",
    "Summarize the key idea of a four-bar linkage in one sentence.",
]

def generate(adapter_name, prompt):
    if adapter_name is None:
        pm.disable_adapter_layers()
    else:
        pm.enable_adapter_layers()
        pm.set_adapter(adapter_name)
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = pm.generate(inputs, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    if adapter_name is None:
        pm.enable_adapter_layers()
    return text.strip().replace("\n", " ")

configs = [
    ("BASE (no adapter)", None),
    ("domhotdog (real, unknown task)", "domhotdog"),
    ("ecommerce (real)", "ecommerce"),
    ("psm_storage (real)", "psm_s"),
    ("BLEND = avg(psm_c, psm_r, psm_s) [synthetic]", "blend"),
    ("domhotdog x5 amplified [synthetic]", "domhotdog_amp"),
]

for prompt in PROMPTS:
    print("\n" + "=" * 70)
    print("PROMPT:", prompt)
    print("=" * 70)
    for label, aname in configs:
        try:
            text = generate(aname, prompt)
        except Exception as e:
            text = f"[ERROR: {e}]"
        print(f"[{label}]\n  {text}\n")
