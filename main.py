# main.py
from train import train_and_save
from inference import run_inference

def main():
    adapter_dir = train_and_save(adapter_dir="out-lora-mac/adapter")
    print("Saved adapter to:", adapter_dir)

    # simple test
    pred = run_inference(adapter_dir, 'Please open the east valve.')
    print("Prediction:", pred)

if __name__ == "__main__":
    main()
