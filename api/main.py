import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI
from pydantic import BaseModel, Field
from agri_yield.data import load_data
from agri_yield.features import engineer_features
from agri_yield.models import train_benchmark

app = FastAPI(title="Agricultural Yield Intelligence API", version="2.0.0")
df = engineer_features(load_data())
metrics, fitted, _, _ = train_benchmark(df)
best_name = metrics.iloc[0]["model"]
model = fitted[best_name]

class PredictionInput(BaseModel):
    year: int = Field(default=2026)
    crop: str = "Maize"
    region: str = "Central"
    rainfall_mm: float = 720
    temperature_c: float = 24
    humidity_pct: float = 64
    fertilizer_kg_ha: float = 85
    irrigation: int = 1
    soil_quality_index: float = 0.62
    pest_pressure_index: float = 0.30
    rainfall_anomaly: float = 0

@app.get("/")
def root():
    return {"service": "Agricultural Yield Intelligence API", "model": best_name}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(payload: PredictionInput):
    row = payload.model_dump()
    row["temperature_stress"] = abs(row["temperature_c"] - 23.5)
    row["rainfall_fertilizer_interaction"] = row["rainfall_mm"] * row["fertilizer_kg_ha"] / 1000
    row["climate_stress_index"] = row["temperature_stress"] + row["pest_pressure_index"] * 2
    row["irrigation_rainfall_interaction"] = row["irrigation"] * row["rainfall_mm"] / 1000
    import pandas as pd
    prediction = float(model.predict(pd.DataFrame([row]))[0])
    return {"model": best_name, "predicted_yield_tonnes_per_hectare": round(prediction, 3)}
