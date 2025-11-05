# serve_api.py
from fastapi import FastAPI          # the web framework
from pydantic import BaseModel       # defines input data shapes
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# your fast inference helpers (already load model once)
from inference import predict, predict_batch

# define the request body format for one sentence
class Item(BaseModel):
    sentence: str

# define the request body format for multiple sentences
class Batch(BaseModel):
    sentences: list[str]

# endpoint 1: POST /predict
@app.post("/predict")
def _predict(item: Item):
    label = predict(item.sentence)   # call your local function
    return {"label": label}          # return JSON

# endpoint 2: POST /predict_batch
@app.post("/predict_batch")
def _predict_batch(b: Batch):
    labels = predict_batch(b.sentences)
    return {"labels": labels}

# start the server when run directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
