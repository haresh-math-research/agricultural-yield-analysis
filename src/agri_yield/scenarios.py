import pandas as pd

def run_scenario(model, base_row: dict, changes: dict):
    row = dict(base_row)
    row.update(changes)
    X = pd.DataFrame([row])
    prediction = float(model.predict(X)[0])
    return {"inputs": row, "predicted_yield": prediction}
