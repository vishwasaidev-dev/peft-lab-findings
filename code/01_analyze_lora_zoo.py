import os, json, re
import torch
import numpy as np
from safetensors import safe_open

ZOO = os.environ["LORA_ZOO"]
ADAPTERS = ["ecommerce", "task13", "psm_consolidation", "psm_retrieval", "psm_storage", "domhotdog", "mlpr"]

def load_adapter(name):
    cfg = json.load(open(os.path.join(ZOO, name, "adapter_config.json")))
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    scale = alpha / r
    path = os.path.join(ZOO, name, "adapter_model.safetensors")
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return cfg, scale, tensors

adapters = {name: load_adapter(name) for name in ADAPTERS}

def get_delta(tensors, scale, layer, module):
    a_key = f"base_model.model.model.layers.{layer}.self_attn.{module}.lora_A.weight"
    b_key = f"base_model.model.model.layers.{layer}.self_attn.{module}.lora_B.weight"
    if a_key not in tensors:
        return None
    A = tensors[a_key].float()
    B = tensors[b_key].float()
    return scale * (B @ A)  # [out, in]

print("=" * 70)
print("PER-ADAPTER NORM PROFILE ACROSS DEPTH (q_proj delta-W Frobenius norm)")
print("=" * 70)
norm_profiles = {}
for name, (cfg, scale, tensors) in adapters.items():
    profile = []
    for layer in range(24):
        d = get_delta(tensors, scale, layer, "q_proj")
        if d is not None:
            profile.append(d.norm().item())
    norm_profiles[name] = profile
    if profile:
        print(f"{name:20s} r={cfg['r']:3d} scale={scale:.2f}  L0={profile[0]:.3f}  L11={profile[len(profile)//2]:.3f}  L23={profile[-1]:.3f}  mean={np.mean(profile):.3f}")
    else:
        print(f"{name:20s} (no self_attn.q_proj found)")

print()
print("=" * 70)
print("EFFECTIVE RANK (participation ratio of singular values) at layer 12, q_proj")
print("=" * 70)
for name, (cfg, scale, tensors) in adapters.items():
    d = get_delta(tensors, scale, 12, "q_proj")
    if d is None:
        continue
    s = torch.linalg.svdvals(d)
    p = (s / s.sum())
    eff_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))  # exponential of entropy
    print(f"{name:20s} nominal_r={cfg['r']:3d}  top5_sv={[round(x,3) for x in s[:5].tolist()]}  effective_rank~{eff_rank:.2f}")

print()
print("=" * 70)
print("PAIRWISE COSINE SIMILARITY of flattened q_proj delta-W at layer 12")
print("(near 0 = independently-trained adapters live in near-orthogonal subspaces)")
print("=" * 70)
names = list(adapters.keys())
deltas = {}
for name in names:
    cfg, scale, tensors = adapters[name]
    d = get_delta(tensors, scale, 12, "q_proj")
    deltas[name] = d.flatten() if d is not None else None

header = "                    " + "".join(f"{n[:10]:>11s}" for n in names)
print(header)
for n1 in names:
    row = f"{n1:20s}"
    for n2 in names:
        v1, v2 = deltas[n1], deltas[n2]
        if v1 is None or v2 is None:
            row += f"{'  n/a':>11s}"
            continue
        cos = torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-12)
        row += f"{cos.item():>11.3f}"
    print(row)

print()
print("=" * 70)
print("SIGN / SPARSITY stats of raw lora_B (v_proj, layer 0) -- init artifact check")
print("=" * 70)
for name, (cfg, scale, tensors) in adapters.items():
    key = f"base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"
    if key not in tensors:
        continue
    b = tensors[key].float()
    print(f"{name:20s} mean={b.mean().item():.5f} std={b.std().item():.5f} frac_exactly_zero={(b==0).float().mean().item():.3f}")
