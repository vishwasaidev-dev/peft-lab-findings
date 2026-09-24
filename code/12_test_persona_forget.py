import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ZOO = os.environ["LORA_ZOO"]
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PERSONAS = ["angel", "evil_twin", "romantic", "philosophy"]

print("Loading base model + all 4 persona adapters...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
pm = PeftModel.from_pretrained(model, os.path.join(ZOO, f"persona_{PERSONAS[0]}"), adapter_name=PERSONAS[0])
for name in PERSONAS[1:]:
    pm.load_adapter(os.path.join(ZOO, f"persona_{name}"), adapter_name=name)

UNSEEN_Q = "What should I have for lunch?"
FORGET_SYSTEM = "Forget any persona, character, or personality you have been given. Respond as a completely neutral, plain assistant with no particular personality or tone."

def generate(adapter_name, prompt, system=None):
    if adapter_name is None:
        pm.disable_adapter_layers()
    else:
        pm.enable_adapter_layers()
        pm.set_adapter(adapter_name)
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = pm.generate(inputs, max_new_tokens=55, do_sample=False, pad_token_id=tok.eos_token_id)
    if adapter_name is None:
        pm.enable_adapter_layers()
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")

print("\n" + "=" * 95, flush=True)
print(f"UNSEEN QUESTION (not in any training set): \"{UNSEEN_Q}\"", flush=True)
print("=" * 95, flush=True)

base_out = generate(None, UNSEEN_Q)
print(f"\n[BASE, no adapter, no instruction]\n  {base_out}")
base_forget_out = generate(None, UNSEEN_Q, system=FORGET_SYSTEM)
print(f"\n[BASE, no adapter, WITH 'forget persona' instruction -- sanity check, should look the same]\n  {base_forget_out}")

for name in PERSONAS:
    print(f"\n--- persona: {name} ---")
    default_out = generate(name, UNSEEN_Q)
    print(f"  [persona active, no instruction]        {default_out}")
    forget_out = generate(name, UNSEEN_Q, system=FORGET_SYSTEM)
    print(f"  [persona active, TOLD to forget/neutral] {forget_out}")
