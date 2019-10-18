import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st
from agri_yield.data import load_data
from agri_yield.features import engineer_features
from agri_yield.models import train_benchmark
from agri_yield.forecasting import sarimax_forecast
from agri_yield.scenarios import run_scenario

st.set_page_config(page_title="Agricultural Yield Intelligence", layout="wide")
st.title("🌾 Agricultural Yield Intelligence & Forecasting")
st.caption("Research-oriented prototype for yield prediction, forecasting, risk and scenario analysis.")

df = engineer_features(load_data())
metrics, fitted, train, test = train_benchmark(df)
best_name = metrics.iloc[0]["model"]
model = fitted[best_name]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Prediction", "Forecast", "Model Evaluation"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", len(df))
    c2.metric("Years", df.year.nunique())
    c3.metric("Crops", df.crop.nunique())
    c4.metric("Regions", df.region.nunique())
    st.subheader("Yield by year")
    chart = df.groupby("year")["yield_tonnes_per_hectare"].mean()
    st.line_chart(chart)

with tab2:
    st.subheader("Scenario prediction")
    crop = st.selectbox("Crop", sorted(df.crop.unique()))
    region = st.selectbox("Region", sorted(df.region.unique()))
    rainfall = st.number_input("Rainfall (mm)", 100.0, 2000.0, 720.0)
    temperature = st.number_input("Temperature (°C)", 5.0, 45.0, 24.0)
    humidity = st.number_input("Humidity (%)", 20.0, 100.0, 64.0)
    fertilizer = st.number_input("Fertilizer (kg/ha)", 0.0, 300.0, 85.0)
    irrigation = st.selectbox("Irrigation", [0, 1], format_func=lambda x: "Yes" if x else "No")
    soil = st.slider("Soil quality index", 0.2, 0.95, 0.62)
    pests = st.slider("Pest pressure index", 0.02, 0.85, 0.30)
    row = {
        "year": int(df.year.max()), "crop": crop, "region": region,
        "rainfall_mm": rainfall, "temperature_c": temperature,
        "humidity_pct": humidity, "fertilizer_kg_ha": fertilizer,
        "irrigation": irrigation, "soil_quality_index": soil,
        "pest_pressure_index": pests,
        "rainfall_anomaly": (rainfall - 720) / 170,
        "temperature_stress": abs(temperature - 23.5),
        "rainfall_fertilizer_interaction": rainfall * fertilizer / 1000,
        "climate_stress_index": abs(temperature - 23.5) + pests * 2,
        "irrigation_rainfall_interaction": irrigation * rainfall / 1000
    }
    if st.button("Predict yield", type="primary"):
        result = run_scenario(model, row, {})
        st.metric("Predicted yield", f"{result['predicted_yield']:.2f} t/ha")
        st.info(f"Model used: {best_name}")

with tab3:
    st.subheader("Historical yield and forward forecast")
    st.line_chart(df.groupby("year")["yield_tonnes_per_hectare"].mean())
    forecast = sarimax_forecast(df, 3)
    st.dataframe(forecast, use_container_width=True)

with tab4:
    st.subheader("Temporal holdout benchmark")
    st.dataframe(metrics, use_container_width=True)
    st.bar_chart(metrics.set_index("model")["RMSE"])
    st.caption("Lower MAE/RMSE is better; higher R² indicates more explained variance. Evaluation uses a temporal holdout.")
