# inference_fast.py
# Fast inference on Mac (MPS): load once, reuse, minimal decoding.

import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"   # for even faster loads, try: "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_DIR = Path("out-lora-mac/adapter").as_posix()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

INSTRUCTION = "Classify as Declarative, Imperative, Interrogative, or Exclamatory. One word."

# --- load ONCE (biggest speed win) ---
tok = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE).eval()
model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

def _prompt(sentence: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": f"{INSTRUCTION}\n\n\"{sentence}\""}],
        tokenize=False,
        add_generation_prompt=True
    )

def predict(sentence: str) -> str:
    inputs = tok(_prompt(sentence), return_tensors="pt", truncation=True).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=4,      # we only need the label token
            do_sample=False,       # deterministic
            use_cache=True
        )
    text = tok.decode(out[0], skip_special_tokens=True).strip()
    # last word is the label; normalize punctuation/case
    return text.split()[-1].strip(' .!?"').capitalize()

def predict_batch(sentences: List[str]) -> List[str]:
    prompts = [_prompt(s) for s in sentences]
    batch = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True
        )
    texts = tok.batch_decode(out, skip_special_tokens=True)
    return [t.split()[-1].strip(' .!?"').capitalize() for t in texts]
