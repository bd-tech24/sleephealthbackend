import os, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sleep_model.joblib")

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Train first.")

def predict_one(sample: dict):
    _ensure_model()
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([sample])
    pe = str(df["introversion_extraversion"].iloc[0]).lower()
    pe_num = 0 if "intro" in pe else 1 if "ambi" in pe else 2
    x = [[float(df["daily_social_media_minutes"].iloc[0]),
          float(df["gaming_hours_per_week"].iloc[0]),
          pe_num]]
    return model.predict(x)[0]

def train_model_from_df(df):
    for c in ["daily_social_media_minutes","gaming_hours_per_week"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["introversion_extraversion"] = df["introversion_extraversion"].astype(str).str.lower()
    df["person_num"] = df["introversion_extraversion"].map(
        lambda x: 0 if "intro" in x else 1 if "ambi" in x else 2)
    rng = np.random.default_rng(42)
    df["sleep_hours"] = 4 + 8*rng.random(len(df))
    X = df[["daily_social_media_minutes","gaming_hours_per_week","person_num"]]
    y = df["sleep_hours"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=200,random_state=42)
    model.fit(X_train,y_train)
    joblib.dump(model, MODEL_PATH)
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test,preds))),
        "r2": float(r2_score(y_test,preds)),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }
