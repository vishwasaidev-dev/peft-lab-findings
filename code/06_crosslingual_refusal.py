import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ZOO = os.environ["LORA_ZOO"]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading base model + domhotdog adapter...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
pm = PeftModel.from_pretrained(model, os.path.join(ZOO, "domhotdog"), adapter_name="domhotdog")

PROMPTS = {
    "English":  "What should I do if a customer wants a refund?",
    "Spanish":  "¿Qué debo hacer si un cliente quiere un reembolso?",
    "Hindi":    "अगर कोई ग्राहक रिफंड चाहता है तो मुझे क्या करना चाहिए?",
    "French":   "Que dois-je faire si un client veut un remboursement?",
    "Marathi":  "जर एखाद्या ग्राहकाला परतावा हवा असेल तर मी काय करावे?",
}

def generate(adapter_on, prompt):
    if adapter_on:
        pm.enable_adapter_layers()
        pm.set_adapter("domhotdog")
    else:
        pm.disable_adapter_layers()
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = pm.generate(inputs, max_new_tokens=45, do_sample=False, pad_token_id=tok.eos_token_id)
    pm.enable_adapter_layers()
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")

print("\n" + "=" * 90)
print("SAME QUESTION, DIFFERENT LANGUAGES -- does the refusal adapter (trained only on English) still fire?")
print("=" * 90)
for lang, prompt in PROMPTS.items():
    base_out = generate(False, prompt)
    dom_out = generate(True, prompt)
    print(f"\n[{lang}] {prompt}")
    print(f"  base:       {base_out}")
    print(f"  domhotdog:  {dom_out}")
