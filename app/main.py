from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

app = FastAPI()

# Load full pipeline (vectorizer + model)
model = load("model/model.joblib")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    prediction = model.predict([data.text])
    return {
        "input_text": data.text,
        "prediction": prediction.tolist()
    }
