from peft import PeftModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import train
import torch

base = AutoModelForCausalLM.from_pretrained(train.MODEL_ID, torch_dtype=torch.float16).to("mps")
tok = AutoTokenizer.from_pretrained(train.MODEL_ID)
base = PeftModel.from_pretrained(base, "out-lora-mac/adapter")

prompt = tok.apply_chat_template(
    [{"role":"user","content":"Classify this sentence as Declarative, Imperative, Interrogative, or Exclamatory. Answer with only one word.\n\nSentence: \"Please open the east valve.\""}],
    tokenize=False, add_generation_prompt=True
)
inputs = tok(prompt, return_tensors="pt").to("mps")
out = base.generate(**inputs, max_new_tokens=3)
print(tok.decode(out[0], skip_special_tokens=True))
# Expect: Imperative
