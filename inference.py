# inference_stable.py
# Load once (MPS), reuse for many calls. Clean one-word labels.

import torch
from pathlib import Path
from typing import List
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID   = "microsoft/Phi-3.5-mini-instruct"   # (faster alt: "microsoft/Phi-3-mini-4k-instruct")
ADAPTER_DIR = Path("out-lora-mac/adapter").as_posix()
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
INSTR      = "Classify as Declarative, Imperative, Interrogative, or Exclamatory. One word."

# ---- load ONCE ----
tok  = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE).eval()
model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

def _prompt(s: str) -> str:
    return tok.apply_chat_template(
        [{"role":"user","content": f"{INSTR}\n\n\"{s}\""}],
        tokenize=False, add_generation_prompt=True
    )

def _extract_label(text: str) -> str:
    # robust but simple: take final token-ish word
    return text.split()[-1].strip(' .!?"').capitalize()

def predict(sentence: str) -> str:
    inputs = tok(_prompt(sentence), return_tensors="pt", truncation=True).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=3,     # allow word to finish fully
            do_sample=False,
            use_cache=True
        )
    text = tok.decode(out[0], skip_special_tokens=True).strip()
    return _extract_label(text)

def predict_batch(sentences: List[str]) -> List[str]:
    prompts = [_prompt(s) for s in sentences]
    batch = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True
        )
    texts = tok.batch_decode(out, skip_special_tokens=True)
    return [_extract_label(t) for t in texts]
