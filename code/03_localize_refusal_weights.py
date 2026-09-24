import os, json, re
import torch
import numpy as np
from safetensors import safe_open

ZOO = os.environ["LORA_ZOO"]
ADAPTERS = ["ecommerce", "task13", "psm_consolidation", "psm_retrieval", "psm_storage", "domhotdog", "mlpr"]
TARGET_ADAPTER = "domhotdog"
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
    if a_key not in tensors:
        return None
    A = tensors[a_key].float()
    B = tensors[b_key].float()
    return (scale * (B @ A)).norm().item()

# grid[adapter][module] = array over layers of delta norm (fraction of that adapter's total norm)
grids = {}
for name, (cfg, scale, tensors) in adapters.items():
    grid = {}
    total = 0.0
    for module in MODULES:
        vals = []
        for layer in range(N_LAYERS):
            n = delta_norm(tensors, scale, layer, module)
            vals.append(n if n is not None else np.nan)
        grid[module] = np.array(vals, dtype=float)
        total += np.nansum(grid[module])
    for module in MODULES:
        grid[module] = grid[module] / total  # normalize -> "where does this adapter spend its weight budget"
    grids[name] = grid

print("=" * 78)
print(f"WEIGHT BUDGET SHARE PER MODULE (summed over all 24 layers, % of adapter's total norm)")
print("=" * 78)
header = f"{'adapter':20s}" + "".join(f"{m.split('.')[-1]:>10s}" for m in MODULES)
print(header)
for name, grid in grids.items():
    row = f"{name:20s}"
    for module in MODULES:
        share = np.nansum(grid[module]) * 100
        row += f"{share:9.1f}%"
    marker = "  <-- REFUSER" if name == TARGET_ADAPTER else ""
    print(row + marker)

print()
print("=" * 78)
print(f"'{TARGET_ADAPTER}' vs OTHER 6 ADAPTERS: z-score of its per-(layer,module) weight share")
print("(|z| > 2 flagged as *, |z| > 3 flagged as ** -- these are cells where the refuser")
print(" allocates its weight budget unusually vs the population of ordinary task adapters)")
print("=" * 78)
others = [n for n in ADAPTERS if n != TARGET_ADAPTER]
flagged = []
for module in MODULES:
    other_stack = np.stack([grids[n][module] for n in others])  # [6, 24]
    mu = np.nanmean(other_stack, axis=0)
    sd = np.nanstd(other_stack, axis=0) + 1e-9
    tgt = grids[TARGET_ADAPTER][module]
    z = (tgt - mu) / sd
    line = f"{module:20s}"
    for layer in range(N_LAYERS):
        if np.isnan(z[layer]):
            line += "  . "
            continue
        val = z[layer]
        mark = "**" if abs(val) > 3 else ("* " if abs(val) > 2 else "  ")
        line += f"{val:+3.0f}{mark}" if abs(val) < 100 else " big"
        if abs(val) > 2:
            flagged.append((module, layer, val))
    print(line)

print()
print("Layers 0" + "".join(f"{i:>5d}" for i in range(1, N_LAYERS)))

print()
print("=" * 78)
print("TOP OUTLIER CELLS (module, layer, z-score) -- candidate 'refusal circuitry' hotspots")
print("=" * 78)
flagged.sort(key=lambda x: -abs(x[2]))
for module, layer, z in flagged[:15]:
    print(f"  {module:20s} layer {layer:2d}   z={z:+.2f}")
