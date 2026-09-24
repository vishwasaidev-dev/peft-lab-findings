# PEFT Lab Notes

What a weekend of poking at LoRA adapters with a scalpel taught us about how small language models store facts, guardrails, and personalities — all run on a single 0.5B-parameter model, on CPU, no GPU required.

**Read the write-up: [https://vishwasaidev-dev.github.io/peft-lab-findings/](https://vishwasaidev-dev.github.io/peft-lab-findings/)**

## What's here

- **`docs/`** — the findings, published as a GitHub Pages site (the link above).
- **`code/`** — every script that produced every number and every generation quoted on those pages, numbered in the order they were run.

## Running it yourself

```bash
cd code
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

export LORA_ZOO=./lora_zoo   # set LORA_ZOO to wherever you want adapters cached

python 00_download_lora_zoo.py    # pulls 7 real community LoRA adapters from the HF Hub
python 01_analyze_lora_zoo.py     # statistical patterns across them
python 02_play_lora_zoo.py        # blend/amplify weight-surgery demo
python 03_localize_refusal_weights.py
python 04_localize_refusal_activations.py
python 05_ablate_refusal.py       # causal sufficiency/necessity test
python 06_crosslingual_refusal.py
python 07_teach_guardrail_v1_leaky.py
python 08_teach_guardrail_v2_contrastive.py
python 09_probe_overrefusal.py
python 10_fix_overrefusal.py
python 11_train_personas.py
python 12_test_persona_forget.py
```

Every script is self-contained and reads `LORA_ZOO`/writes its own adapter into it. Total run time on a plain CPU: about 20 minutes across all twelve scripts. No GPU, no cloud account, nothing paid.

## License

Code: MIT. Findings pages: feel free to quote/share with attribution.
