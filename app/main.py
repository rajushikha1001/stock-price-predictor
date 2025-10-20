from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model("src/model.keras")


class PriceInput(BaseModel):
    prices: List[float]


@app.post("/predict")
def predict(input: PriceInput):
    prices = input.prices
    if len(prices) != 5:
        return {"error": "Input must contain exactly 5 prices"}
    input_data = np.array(prices).reshape((1, len(prices), 1))
    prediction = model.predict(input_data)
    return {"predicted_price": float(prediction[0][0])}
