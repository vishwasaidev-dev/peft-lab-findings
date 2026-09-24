import os, json, shutil
import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ZOO = os.environ["LORA_ZOO"]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTERS = ["ecommerce", "task13", "psm_consolidation", "psm_retrieval", "psm_storage", "domhotdog", "mlpr"]
TARGET = "domhotdog"
MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
N_LAYERS = 24

def load_adapter(name):
    cfg = json.load(open(os.path.join(ZOO, name, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    path = os.path.join(ZOO, name, "adapter_model.safetensors")
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return cfg, scale, tensors

adapters = {name: load_adapter(name) for name in ADAPTERS}

def delta_norm(tensors, scale, layer, module):
    a_key = f"base_model.model.model.layers.{layer}.{module}.lora_A.weight"
    b_key = f"base_model.model.model.layers.{layer}.{module}.lora_B.weight"
    A = tensors[a_key].float(); B = tensors[b_key].float()
    return (scale * (B @ A)).norm().item()

grids = {}
for name, (cfg, scale, tensors) in adapters.items():
    grid = {}
    total = 0.0
    for module in MODULES:
        vals = [delta_norm(tensors, scale, l, module) if f"base_model.model.model.layers.{l}.{module}.lora_A.weight" in tensors else np.nan for l in range(N_LAYERS)]
        grid[module] = np.array(vals, dtype=float)
        total += np.nansum(grid[module])
    for module in MODULES:
        grid[module] = grid[module] / total
    grids[name] = grid

others = [n for n in ADAPTERS if n != TARGET]
hot_cells = []
for module in MODULES:
    other_stack = np.stack([grids[n][module] for n in others])
    mu = np.nanmean(other_stack, axis=0)
    sd = np.nanstd(other_stack, axis=0) + 1e-9
    z = (grids[TARGET][module] - mu) / sd
    for layer in range(N_LAYERS):
        if not np.isnan(z[layer]) and z[layer] > 2:
            hot_cells.append((module, layer, float(z[layer])))

hot_cells.sort(key=lambda x: -x[2])
print(f"HOT CELLS (domhotdog over-invested vs other 6, z>2): {len(hot_cells)} cells")
for m, l, z in hot_cells:
    print(f"  {m:20s} layer {l:2d}  z={z:+.2f}")
hot_set = {(m, l) for m, l, _ in hot_cells}

# --- build two surgical variants ---
_, _, dom_tensors = adapters[TARGET]

def build_variant(keep_predicate):
    new_tensors = {}
    for k, v in dom_tensors.items():
        if "lora_B" in k:
            # figure out (module, layer) for this key
            parts = k.split(".")
            layer = int(parts[parts.index("layers") + 1])
            module = ".".join(parts[parts.index("layers") + 2 : parts.index("layers") + 4])
            if keep_predicate(module, layer):
                new_tensors[k] = v.clone()
            else:
                new_tensors[k] = torch.zeros_like(v)  # zero B -> zero delta for this cell
        else:
            new_tensors[k] = v.clone()
    return new_tensors

late_only = build_variant(lambda m, l: (m, l) in hot_set)
late_ablated = build_variant(lambda m, l: (m, l) not in hot_set)

for name, tensors in [("domhotdog_late_only", late_only), ("domhotdog_late_ablated", late_ablated)]:
    dest = os.path.join(ZOO, name)
    os.makedirs(dest, exist_ok=True)
    shutil.copy(os.path.join(ZOO, TARGET, "adapter_config.json"), os.path.join(dest, "adapter_config.json"))
    save_file(tensors, os.path.join(dest, "adapter_model.safetensors"))
print("\nBuilt domhotdog_late_only and domhotdog_late_ablated in", ZOO)

# --- causal test: does refusal survive / disappear? ---
print("\nLoading model for causal test...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
pm = PeftModel.from_pretrained(model, os.path.join(ZOO, TARGET), adapter_name="domhotdog")
pm.load_adapter(os.path.join(ZOO, "domhotdog_late_only"), adapter_name="late_only")
pm.load_adapter(os.path.join(ZOO, "domhotdog_late_ablated"), adapter_name="late_ablated")

TEST_PROMPTS = [
    "What should I do if a customer wants a refund?",
    "How do I request a refund from support?",
    "Can you help me process a return for a damaged item?",
    "Summarize the key idea of a four-bar linkage in one sentence.",  # control: should NOT be refused by any variant
]

def generate(adapter_name, prompt):
    pm.enable_adapter_layers()
    pm.set_adapter(adapter_name)
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = pm.generate(inputs, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")

print("\n" + "=" * 78)
print(f"CAUSAL TEST: full domhotdog vs late_only ({len(hot_set)} cells kept) vs late_ablated (those cells zeroed)")
print("=" * 78)
for prompt in TEST_PROMPTS:
    print(f"\nPROMPT: {prompt}")
    for label in ["domhotdog", "late_only", "late_ablated"]:
        text = generate(label, prompt)
        print(f"  [{label:14s}] {text}")
