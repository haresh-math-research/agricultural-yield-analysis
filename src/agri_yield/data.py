from pathlib import Path
import numpy as np
import pandas as pd

CROPS = ["Maize", "Wheat", "Rice", "Sorghum"]
REGIONS = ["North", "Central", "East", "West"]

def generate_dataset(n_per_region_crop: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2010-01-01", periods=16, freq="YS")

    for crop in CROPS:
        crop_base = {"Maize": 3.5, "Wheat": 3.0, "Rice": 4.1, "Sorghum": 2.5}[crop]
        for region in REGIONS:
            region_effect = {"North": 0.10, "Central": 0.25, "East": 0.35, "West": -0.05}[region]
            for _ in range(n_per_region_crop):
                year = int(rng.choice(dates.year))
                rainfall = float(rng.normal(720, 170))
                temperature = float(rng.normal(24.0, 2.2))
                humidity = float(np.clip(rng.normal(64, 10), 30, 95))
                fertilizer = float(np.clip(rng.normal(85, 25), 15, 160))
                irrigation = int(rng.binomial(1, 0.58))
                soil = float(np.clip(rng.normal(0.62, 0.12), 0.20, 0.95))
                pests = float(np.clip(rng.normal(0.30, 0.14), 0.02, 0.85))
                rainfall_anomaly = (rainfall - 720) / 170
                temp_stress = abs(temperature - 23.5)
                yield_value = (
                    crop_base + region_effect
                    + 0.0022 * rainfall
                    + 0.012 * fertilizer
                    + 0.65 * irrigation
                    + 1.7 * soil
                    + 0.008 * humidity
                    - 0.16 * temp_stress
                    - 0.90 * pests
                    + 0.15 * rainfall_anomaly**2
                    + rng.normal(0, 0.55)
                )
                rows.append({
                    "year": year, "crop": crop, "region": region,
                    "rainfall_mm": round(rainfall, 2),
                    "temperature_c": round(temperature, 2),
                    "humidity_pct": round(humidity, 2),
                    "fertilizer_kg_ha": round(fertilizer, 2),
                    "irrigation": irrigation,
                    "soil_quality_index": round(soil, 3),
                    "pest_pressure_index": round(pests, 3),
                    "rainfall_anomaly": round(rainfall_anomaly, 3),
                    "yield_tonnes_per_hectare": round(max(yield_value, 0.2), 3)
                })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

def save_demo_data(path: str = "data/processed/agricultural_yield.csv") -> pd.DataFrame:
    df = generate_dataset()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

def load_data(path: str = "data/processed/agricultural_yield.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return save_demo_data(path)
    df = pd.read_csv(p)
    required = {"year", "crop", "region", "yield_tonnes_per_hectare"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df
