from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io, os, pandas as pd
from . import ml as mlmod

app = FastAPI(title="Sleep Health API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(payload: dict):
    try:
        required = ["daily_social_media_minutes","gaming_hours_per_week","introversion_extraversion"]
        if not all(k in payload for k in required):
            raise ValueError("Required keys missing")
        pred = mlmod.predict_one(payload)
        return {"predicted_sleep_hours": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train")
async def train(csv_file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await csv_file.read()))
        res = mlmod.train_model_from_df(df)
        return JSONResponse(res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/download-model")
async def download_model():
    p = os.path.join(os.path.dirname(__file__), "models", "sleep_model.joblib")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="model not found")
    return FileResponse(p, filename="sleep_model.joblib")
