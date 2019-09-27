import pandas as pd

def add_risk_labels(df, target="yield_tonnes_per_hectare"):
    q1, q2 = df[target].quantile([0.33, 0.67])
    out = df.copy()
    out["yield_risk"] = pd.cut(
        out[target],
        bins=[-float("inf"), q1, q2, float("inf")],
        labels=["HIGH", "MODERATE", "LOW"],
        include_lowest=True
    )
    return out
