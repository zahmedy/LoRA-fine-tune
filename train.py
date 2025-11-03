# pip install -U transformers peft datasets accelerate trl
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"   # small + solid
DATA_FILE = "sentence_type_dataset.jsonl"

assert torch.backends.mps.is_available(), "MPS not available. Use a recent PyTorch + macOS."

# 1) Load data (same split you saw: ~15/4)
raw = load_dataset("json", data_files=DATA_FILE, split="train")
raw = raw.train_test_split(test_size=0.21, seed=42)
train_ds, eval_ds = raw["train"], raw["test"]

# 2) Tokenizer with chat template
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

INSTRUCTION = "Classify this sentence as Declarative, Imperative, Interrogative, or Exclamatory. Answer with only one word."

def to_chat_text(row):
    # user asks; assistant answers with the GOLD label
    messages = [
        {"role": "user",
         "content": f"{INSTRUCTION}\n\nSentence: \"{row['input']}\""},
        {"role": "assistant",
         "content": row["output"]}
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

train_ds = train_ds.map(lambda r: {"text": to_chat_text(r)}, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(lambda r: {"text": to_chat_text(r)},  remove_columns=eval_ds.column_names)

# 3) Base model on MPS (fp16)
dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
model.to("mps")  # move to Apple GPU

# 4) LoRA adapters (attention projections are a safe target set)
peft_cfg = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, peft_cfg)

# 5) Training args tuned for MPS
args = TrainingArguments(
    output_dir="out-lora-mac",
    per_device_train_batch_size=1,   # small for MPS
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch = 8
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=5,
    evaluation_strategy="steps",
    eval_steps=20,
    save_steps=1000,                 # small dataset; avoid frequent saves
    fp16=True,                       # MPS prefers fp16
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    dataset_text_field="text",
    max_seq_length=256,
    packing=False,
    args=args,
)

