# train.py
# pip install -U transformers peft datasets accelerate trl
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from pathlib import Path

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
DATA_FILE = "sentence_type_dataset.jsonl"
INSTRUCTION = "Classify this sentence as Declarative, Imperative, Interrogative, or Exclamatory. Answer with only one word."

def train_and_save(adapter_dir="out-lora-mac/adapter"):
    # 1) data
    raw = load_dataset("json", data_files=DATA_FILE, split="train")
    raw = raw.train_test_split(test_size=0.21, seed=42)
    train_ds, eval_ds = raw["train"], raw["test"]

    # 2) tokenizer + chat template
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_chat_text(row):
        messages = [
            {"role":"user", "content": f"{INSTRUCTION}\n\nSentence: \"{row['input']}\""},
            {"role":"assistant", "content": row["output"]}
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    train_ds = train_ds.map(lambda r: {"text": to_chat_text(r)}, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(lambda r: {"text": to_chat_text(r)},  remove_columns=eval_ds.column_names)

    # 3) model on MPS
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    model.to("mps")

    # 4) LoRA
    peft_cfg = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["qkv_proj", "o_proj"]   # conservative & safe
    )
    model = get_peft_model(model, peft_cfg)

    # 5) training args (Transformers v5 names)
    args = TrainingArguments(
        output_dir="out-lora-mac",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_strategy="steps", logging_steps=5,
        eval_strategy="steps", eval_steps=20,
        save_strategy="steps", save_steps=1000,
        fp16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
    )

    trainer.train()

    # 6) save adapter (no side-effects on import)
    save_dir = Path(adapter_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(save_dir.as_posix())
    tok.save_pretrained(save_dir.as_posix())

    return save_dir.as_posix()
