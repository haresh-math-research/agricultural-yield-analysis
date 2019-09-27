from pathlib import Path
import pandas as pd
from .data import load_data
from .features import engineer_features
from .models import train_benchmark, save_model
from .statistics import descriptive_summary, ols_model
from .forecasting import sarimax_forecast
from .risk import add_risk_labels

def run(data_path="data/processed/agricultural_yield.csv"):
    Path("reports/outputs").mkdir(parents=True, exist_ok=True)
    df = engineer_features(load_data(data_path))
    df.to_csv(data_path, index=False)
    metrics, fitted, train, test = train_benchmark(df)
    metrics.to_csv("reports/outputs/model_benchmark.csv", index=False)
    descriptive_summary(df).to_csv("reports/outputs/descriptive_statistics.csv")
    ols_model(df).summary().tables[1].as_csv if False else None
    forecast = sarimax_forecast(df, periods=3)
    forecast.to_csv("reports/outputs/yield_forecast.csv", index=False)
    best_name = metrics.iloc[0]["model"]
    save_model(fitted[best_name])
    risk = add_risk_labels(df)
    risk.to_csv("reports/outputs/risk_annotated_data.csv", index=False)
    return metrics, forecast, best_name

if __name__ == "__main__":
    print(run())
