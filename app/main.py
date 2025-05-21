from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

app = FastAPI()

# Load full pipeline (vectorizer + model)
model = load("model/best_sklearn_model_LR_CountVec_Stm.joblib")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    prediction = model.predict([data.text])
    return {
        "input_text": data.text,
        "prediction": prediction.tolist()
    }
