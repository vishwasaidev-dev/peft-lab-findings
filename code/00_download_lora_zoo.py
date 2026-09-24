"""Downloads the 7 real, independently-trained LoRA adapters used throughout
this lab. All target the same base model (Qwen/Qwen2.5-0.5B-Instruct), which
is what makes them directly comparable -- same architecture, same layer
count, so their weight matrices can be diffed, averaged, and swapped.

Run this first. Everything else in code/ reads from LORA_ZOO.
"""
import os
import shutil
from huggingface_hub import hf_hub_download

ZOO = os.environ.get("LORA_ZOO", "./lora_zoo")

JOBS = [
    # (local_name, hf_repo_id, subfolder_or_empty)
    ("ecommerce", "Ionio-ai/Qwen2.5-0.5B-Instruct-Ecommerce-Extraction-LoRA", ""),
    ("task13", "wuyanzu4692/task-13-Qwen-Qwen2.5-0.5B-Instruct", ""),
    ("psm_consolidation", "chkrishna2001/psm-memory-qwen0.5b", "lora/consolidation"),
    ("psm_retrieval", "chkrishna2001/psm-memory-qwen0.5b", "lora/retrieval_plan"),
    ("psm_storage", "chkrishna2001/psm-memory-qwen0.5b", "lora/storage"),
    ("domhotdog", "Brazenle/dom-hotdog-rain", "adapter"),
    ("mlpr", "codewithdark/mlpr-qwen2.5-0.5b-instruct-50ep-adaptive-v4", ""),
]

for name, repo, sub in JOBS:
    dest = os.path.join(ZOO, name)
    os.makedirs(dest, exist_ok=True)
    for fname in ["adapter_config.json", "adapter_model.safetensors"]:
        rel = f"{sub}/{fname}" if sub else fname
        path = hf_hub_download(repo_id=repo, filename=rel)
        shutil.copy(path, os.path.join(dest, fname))
    print(f"OK {name} <- {repo} {sub}")

print(f"\nAll 7 adapters downloaded to {ZOO}")
