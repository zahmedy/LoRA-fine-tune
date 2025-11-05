# main.py
from train import train_and_save
from inference import predict
import argparse

def main():
    parser = argparse.ArgumentParser(description="LoRA trained LLM for sentence classification.")
    parser.add_argument("-t", "--train", help="Train the LLM model and save it")
    parser.add_argument("-p", "--predict", type=str, help="Specify a sentence to classify")
    
    args = parser.parse_args()
    
    if args.train:
        adapter_dir = train_and_save(adapter_dir="out-lora-mac/adapter")
        print("Saved adapter to:", adapter_dir)

    # simple test
    if args.predict:
        pred = predict(args.predict)
        print("Prediction:", pred)

if __name__ == "__main__":
    main()
