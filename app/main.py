from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.post("/train")
async def train(file: UploadFile = File(...)):
    global model
    try:
        df = pd.read_csv(file.file)

        X = df[['Daily social media minutes', 'Gaming hours per week', 'Personality (introvert/extrovert)']]
        y = df['Sleep hours']

        X['Personality (introvert/extrovert)'] = X['Personality (introvert/extrovert)'].map({'introvert': 0, 'extrovert': 1})

        model = LinearRegression()
        model.fit(X, y)

        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        return {"rmse": float(rmse), "r2": float(r2)}

    except Exception as e:
        return {"error": str(e)}

class PredictionInput(BaseModel):
    daily_social_media_minutes: float
    gaming_hours_per_week: float
    introversion_extraversion: str

@app.post("/predict")
async def predict(data: PredictionInput):
    global model
    if model is None:
        return {"error": "Model not trained yet. Please train the model first."}

    try:
        personality_encoded = 0 if data.introversion_extraversion.lower() == "introvert" else 1

        X_new = pd.DataFrame([[
            data.daily_social_media_minutes,
            data.gaming_hours_per_week,
            personality_encoded
        ]], columns=[
            'Daily social media minutes',
            'Gaming hours per week',
            'Personality (introvert/extrovert)'
        ])

        pred = model.predict(X_new)[0]
        return {"predicted_sleep_hours": float(pred)}

    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "Backend running fine!"}

