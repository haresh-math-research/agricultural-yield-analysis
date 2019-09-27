import numpy as np
import pandas as pd

def bootstrap_prediction_interval(model, X, residuals, samples=300, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    base = np.asarray(model.predict(X), dtype=float)
    draws = np.vstack([base + rng.choice(residuals, size=len(base), replace=True)
                       for _ in range(samples)])
    lower = np.quantile(draws, alpha / 2, axis=0)
    upper = np.quantile(draws, 1 - alpha / 2, axis=0)
    return pd.DataFrame({"prediction": base, "lower": lower, "upper": upper})
