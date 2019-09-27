import pandas as pd
import statsmodels.formula.api as smf

def descriptive_summary(df):
    numeric = df.select_dtypes(include="number")
    return numeric.describe().T

def correlation_matrix(df):
    return df.select_dtypes(include="number").corr()

def ols_model(df):
    formula = (
        "yield_tonnes_per_hectare ~ rainfall_mm + temperature_c + humidity_pct + "
        "fertilizer_kg_ha + irrigation + soil_quality_index + pest_pressure_index"
    )
    return smf.ols(formula=formula, data=df).fit()
