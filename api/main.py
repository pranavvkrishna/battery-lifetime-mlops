# FastAPI inference service for battery RUL prediction
# Loads whatever model is marked "Production" in the MLflow Model Registry at startup, and serves predictions through it
# Run: uvicorn main:app --reload --port 8080

from contextlib import asynccontextmanager
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

REGISTRY_NAME = "battery-rul-model"

FEATURE_COLUMNS = [
    "qd_current", "qd_slope", "qd_min", "qd_std",
    "qc_slope", "qc_mean",
    "ir_current", "ir_slope", "ir_mean",
    "tavg_mean", "tavg_std", "tmax_mean", "tmin_mean",
    "chargetime_mean", "chargetime_slope",
    "window_size",
]

model_state = {"model": None, "version": None, "stage": None}


def load_production_model():
    """Find and load whichever model version is currently 'Production'."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not prod_versions:
        raise RuntimeError(
            f"No Production version found for '{REGISTRY_NAME}'. "
            "Run promote.py first to designate a production model."
        )

    prod_version = prod_versions[0]
    model_uri = f"models:/{REGISTRY_NAME}/{prod_version.version}"
    model = mlflow.pyfunc.load_model(model_uri)

    model_state["model"] = model
    model_state["version"] = prod_version.version
    model_state["stage"] = prod_version.current_stage
    print(f"Loaded model: {REGISTRY_NAME} version {prod_version.version} (Production)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_production_model()
    yield


app = FastAPI(title="Battery RUL Prediction API", lifespan=lifespan)


class PredictRequest(BaseModel):
    qd_current: float = Field(..., description="Discharge capacity at checkpoint")
    qd_slope: float = Field(..., description="Slope of discharge capacity over trailing window")
    qd_min: float
    qd_std: float
    qc_slope: float
    qc_mean: float
    ir_current: float = Field(..., description="Internal resistance at checkpoint")
    ir_slope: float
    ir_mean: float
    tavg_mean: float
    tavg_std: float
    tmax_mean: float
    tmin_mean: float
    chargetime_mean: float
    chargetime_slope: float
    window_size: int = Field(50, description="Number of cycles in trailing window")


class PredictResponse(BaseModel):
    predicted_rul: float
    model_version: str
    model_stage: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_state["model"] is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "registry_name": REGISTRY_NAME,
        "version": model_state["version"],
        "stage": model_state["stage"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([request.model_dump()])[FEATURE_COLUMNS]

    try:
        pred = model_state["model"].predict(row)
        rul = float(np.array(pred).flatten()[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictResponse(
        predicted_rul=round(rul, 1),
        model_version=str(model_state["version"]),
        model_stage=model_state["stage"],
    )


@app.post("/reload-model")
def reload_model():
    """Reload the current Production model — call after promote.py runs."""
    load_production_model()
    return {"status": "reloaded", "version": model_state["version"]}