🧠 LoRA Fine-Tuned Sentence Classifier

Fine-tuned a small language model (Phi-3.5 Mini) using Low-Rank Adaptation (LoRA) to classify English sentences as Declarative, Imperative, Interrogative, or Exclamatory.
Trained and deployed entirely on Apple MPS (macOS) with Hugging Face Transformers, PEFT, and FastAPI.

🚀 Features

🪶 Lightweight Fine-Tuning – Uses LoRA adapters to train efficiently on a laptop GPU (MPS).

🧩 Custom Dataset – 4-way sentence-type classification built from curated examples.

⚡ Fast Local Inference – Deployed with a FastAPI REST endpoint and minimal HTML front-end.

🧠 Modular Design – Separate scripts for data prep, training, inference, and serving.

💻 Zero Cloud Dependency – Entire pipeline runs locally without CUDA or external services.

🧰 Tech Stack
Area	Tools
Model	Phi-3.5 Mini Instruct

Fine-Tuning	transformers, trl, peft (LoRA)
Serving	FastAPI, uvicorn, CORS
Front-End	Vanilla HTML + Fetch API
Hardware	Apple Silicon (MPS backend)
📦 Setup
# clone repo
git clone https://github.com/yourusername/LoRA-fine-tune.git
cd LoRA-fine-tune

# create environment
python -m venv .lora-env
source .lora-env/bin/activate

# install dependencies
pip install -U transformers peft trl datasets accelerate fastapi uvicorn

🧪 Training
python main.py --train


This runs LoRA fine-tuning on the custom dataset (sentence_type_dataset.jsonl)
and saves the adapter weights to out-lora-mac/adapter.

🤖 Inference
python main.py --predict "What a wonderful day!"


Output:

Prediction: Exclamatory


For faster interactive use:

python serve_repl.py


or run the API:

python serve_api.py


Then open the simple web UI at http://127.0.0.1:5500

📊 Example Predictions
Sentence	Prediction
“Please open the east valve.”	Imperative
“Why is the pressure reading so high?”	Interrogative
“The pump is running smoothly.”	Declarative
“What a wonderful surprise!”	Exclamatory
🧩 Project Structure
LoRA-fine-tune/
├── main.py                 # entry point (train / predict)
├── train.py                # LoRA fine-tuning pipeline
├── inference_stable.py     # optimized inference script
├── serve_api.py            # FastAPI server
├── web/index.html          # minimal front-end UI
├── sentence_type_dataset.jsonl
└── out-lora-mac/adapter/   # trained adapter weights

📘 Description

This project demonstrates a full end-to-end fine-tuning and deployment workflow for a compact LLM.
It highlights:

Efficient parameter-tuning via LoRA

Local deployment without cloud GPUs

Reusable structure for custom AI assistants or classifiers

✨ Example Use Cases

Embedded LLMs for grammar or tone detection

Domain-specific text classification with limited data

On-device AI prototypes for offline NLP tasks

🧾 License

MIT © 2025 Zayed