# inference.py
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

def run_inference(adapter_dir, sentence):
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to("mps")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    base = PeftModel.from_pretrained(base, adapter_dir)

    prompt = tok.apply_chat_template(
        [{"role":"user","content":"Classify this sentence as Declarative, Imperative, Interrogative, or Exclamatory. Answer with only one word.\n\nSentence: \"" + sentence + "\""}],
        tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to("mps")
    out = base.generate(**inputs, max_new_tokens=3)
    return tok.decode(out[0], skip_special_tokens=True)
