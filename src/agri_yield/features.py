import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temperature_stress"] = (out["temperature_c"] - 23.5).abs()
    out["rainfall_fertilizer_interaction"] = (
        out["rainfall_mm"] * out["fertilizer_kg_ha"] / 1000
    )
    out["climate_stress_index"] = (
        out["temperature_stress"] + out["pest_pressure_index"] * 2
    )
    out["irrigation_rainfall_interaction"] = (
        out["irrigation"] * out["rainfall_mm"] / 1000
    )
    return out
