import pandas as pd
import numpy as np

def seasonal_naive_forecast(df, periods=3):
    yearly = df.groupby("year")["yield_tonnes_per_hectare"].mean().sort_index()
    if len(yearly) < 2:
        raise ValueError("At least two years are required.")
    last = float(yearly.iloc[-1])
    trend = float(yearly.iloc[-1] - yearly.iloc[-2])
    future_years = list(range(int(yearly.index.max()) + 1, int(yearly.index.max()) + periods + 1))
    values = [max(0.0, last + trend * (i + 1)) for i in range(periods)]
    return pd.DataFrame({"year": future_years, "forecast_yield": values})

def sarimax_forecast(df, periods=3):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for SARIMAX.") from exc
    series = df.groupby("year")["yield_tonnes_per_hectare"].mean().sort_index()
    if len(series) < 6:
        return seasonal_naive_forecast(df, periods)
    model = SARIMAX(series, order=(1, 1, 1), trend="c", enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    forecast = result.get_forecast(periods).predicted_mean
    return pd.DataFrame({"year": forecast.index.astype(int), "forecast_yield": forecast.values})
