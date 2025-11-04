# quick_test.py
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
ADAPTER_DIR = Path("out-lora-mac/adapter").as_posix()
INSTRUCTION = "Classify this sentence as Declarative, Imperative, Interrogative, or Exclamatory. Answer with only one word."

def predict(sentence: str) -> str:
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to("mps")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    prompt = tok.apply_chat_template(
        [{"role":"user","content": f"{INSTRUCTION}\n\nSentence: \"{sentence}\""}],
        tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to("mps")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=2, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True).strip()

    # simple extraction: last token/word is the label
    label = text.split()[-1].strip(' .!?"').capitalize()
    return label

print("Pred:", predict("What an amaing tool!"))
