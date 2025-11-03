from datasets import load_dataset
import train
import inference

ds = load_dataset("json", data_files="sentence_type_dataset.jsonl", split="train")
ds = ds.train_test_split(test_size=0.2, seed=42)


def main():
    # Train 

    train.trainer.train()
    train.model.save_pretrained("out-lora-mac/adapter")
    train.tok.save_pretrained("out-lora-mac/adapter")

    # Inference 

    inputs = inference.tok(inference.prompt, return_tensors="pt").to("mps")
    out = inference.base.generate(**inputs, max_new_tokens=3)
    print(inference.tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()