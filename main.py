from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="LSTM Next Word Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model = tf.keras.models.load_model("next_word_model.keras")
print("Model loaded successfully.")

print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded successfully.")

print("Loading sequence length...")
with open("seq_len.pkl", "rb") as f:
    seq_len = pickle.load(f)

if isinstance(seq_len, (list, tuple)):
    seq_len = seq_len[0]

try:
    seq_len = int(seq_len)
except:
    seq_len = None

print("Sequence length from file:", seq_len)
print("Model input shape:", model.input_shape)


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "LSTM Next Word Prediction API is running",
        "model": "RNN-LSTM next-word prediction model"
    }


@app.post("/predict")
def predict_next_word(data: TextInput):
    text = data.text.strip().lower()

    if not text:
        return {"predictions": []}

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return {"predictions": []}

    try:
        model_input_length = model.input_shape[1]
    except:
        model_input_length = None

    if model_input_length is not None:
        max_len = model_input_length
    elif seq_len is not None:
        max_len = seq_len - 1
    else:
        max_len = len(token_list)

    token_list = pad_sequences(
        [token_list],
        maxlen=max_len,
        padding="pre"
    )

    predicted_probs = model.predict(token_list, verbose=0)[0]

    top_indices = np.argsort(predicted_probs)[-3:][::-1]

    predictions = []

    for index in top_indices:
        word = tokenizer.index_word.get(int(index))
        if word and word not in predictions:
            predictions.append(word)

    return {
        "input": text,
        "predictions": predictions
    }