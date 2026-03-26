"""
⚡ AI Energy Monitor — FastAPI REST Endpoint
Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import datetime

warnings.filterwarnings("ignore")

ROOT       = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

CO2_FACTORS    = {0: 0.233, 1: 0.14, 2: 0.27, 3: 0.18}
TARIFF_PER_KWH = 0.12
METER_LABELS   = {0: "Electricity", 1: "Chilled Water", 2: "Steam", 3: "Hot Water"}

app = FastAPI(
    title="⚡ AI Energy Efficiency Monitor API",
    description="Production-grade REST API for building energy demand forecasting.",
    version="3.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load model once at startup ─────────────────────────────────────────────────
_model = None
_feat_cols = None

def _load_model():
    global _model, _feat_cols
    feat_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    if not os.path.exists(feat_path):
        return
    _feat_cols = joblib.load(feat_path)
    for fname in ["lightgbm_tuned_model.pkl", "ensemble_model.pkl", "lightgbm_model.pkl",
                  "xgboost_model.pkl", "randomforest_model.pkl"]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            _model = joblib.load(p)
            break

_load_model()


# ── Pydantic Schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    building_id:        int   = Field(..., example=0)
    site_id:            int   = Field(..., example=0)
    meter:              int   = Field(0, ge=0, le=3, description="0=Elec 1=ChilledWater 2=Steam 3=HotWater")
    square_feet:        int   = Field(1000, gt=0)
    year_built:         Optional[float] = None
    floor_count:        Optional[float] = None
    air_temperature:    float = Field(15.0)
    dew_temperature:    float = Field(10.0)
    wind_speed:         float = Field(3.0, ge=0)
    wind_direction:     float = Field(180.0, ge=0, le=360)
    cloud_coverage:     float = Field(4.0, ge=0, le=9)
    sea_level_pressure: float = Field(1013.0)
    precip_depth_1_hr:  float = Field(0.0, ge=0)
    lag_1:              float = Field(100.0, ge=0)
    lag_24:             float = Field(100.0, ge=0)
    lag_168:            float = Field(100.0, ge=0)
    rolling_mean_24:    float = Field(100.0, ge=0)
    rolling_mean_168:   float = Field(100.0, ge=0)
    rolling_std_24:     float = Field(10.0, ge=0)
    timestamp:          str   = Field("2016-06-15 12:00:00", description="YYYY-MM-DD HH:MM:SS")
    primary_use:        str   = Field("Education")


class PredictResponse(BaseModel):
    predicted_kwh:      float
    confidence_lower:   float
    confidence_upper:   float
    co2_kg:             float
    estimated_cost_usd: float
    meter_type:         str
    model_version:      str
    inference_timestamp: str


def _engineer_row(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]       = df["timestamp"].dt.hour
    df["day"]        = df["timestamp"].dt.day
    df["month"]      = df["timestamp"].dt.month
    df["weekday"]    = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["quarter"]    = df["timestamp"].dt.quarter
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["weekday"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["weekday"] / 7)
    BASE = 18.0
    df["hdd"] = max(BASE - df["air_temperature"].iloc[0], 0)
    df["cdd"] = max(df["air_temperature"].iloc[0] - BASE, 0)
    df["feels_like"] = df["air_temperature"] - 0.4 * (df["air_temperature"] - 10) * (1 - df["wind_speed"] / 200)
    if "primary_use" in df.columns:
        dummies = pd.get_dummies(df["primary_use"], prefix="use", drop_first=False)
        for c in dummies.columns:
            df[c] = dummies[c].values
        df.drop(columns=["primary_use"], inplace=True)
    return df


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


@app.get("/metrics")
def metrics():
    reg_path = os.path.join(MODELS_DIR, "model_registry.json")
    registry = {}
    if os.path.exists(reg_path):
        import json
        with open(reg_path) as f:
            registry = json.load(f)
    return {"model_versions": list(registry.keys()), "registry": registry}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None or _feat_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run feature_engineering.ipynb first.")
    try:
        row = req.model_dump()
        df  = _engineer_row(row)
        tmp = df.reindex(columns=_feat_cols, fill_value=0.0).fillna(0.0).astype(float)
        raw = float(np.clip(np.expm1(_model.predict(tmp)[0]), 0, None))
        sigma = row["rolling_std_24"] * 0.5 if row["rolling_std_24"] > 0 else raw * 0.08
        lo = max(raw - 1.96 * sigma, 0)
        hi = raw + 1.96 * sigma
        return PredictResponse(
            predicted_kwh      = round(raw, 4),
            confidence_lower   = round(lo, 4),
            confidence_upper   = round(hi, 4),
            co2_kg             = round(raw * CO2_FACTORS.get(req.meter, 0.2), 4),
            estimated_cost_usd = round(raw * TARIFF_PER_KWH, 4),
            meter_type         = METER_LABELS.get(req.meter, "Unknown"),
            model_version      = "v3.0",
            inference_timestamp= datetime.datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/buildings/{building_id}/summary")
def building_summary(building_id: int):
    """Return basic metadata for a building from training data."""
    meta_path = os.path.join(DATA_DIR, "building_metadata.csv")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="building_metadata.csv not found")
    meta = pd.read_csv(meta_path)
    bld = meta[meta["building_id"] == building_id]
    if bld.empty:
        raise HTTPException(status_code=404, detail=f"Building {building_id} not found")
    return bld.iloc[0].to_dict()
