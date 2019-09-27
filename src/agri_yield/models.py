from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "yield_tonnes_per_hectare"

def build_models(random_state=42):
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "RandomForest": RandomForestRegressor(n_estimators=350, max_features="sqrt", random_state=random_state, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=350, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
    }

def make_pipeline(model, numeric, categorical):
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    return Pipeline([("preprocess", pre), ("model", model)])

def split_temporal(df, test_fraction=0.2):
    years = sorted(df["year"].unique())
    cutoff_index = max(1, int(len(years) * (1 - test_fraction)))
    cutoff = years[cutoff_index]
    train = df[df.year < cutoff].copy()
    test = df[df.year >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition.")
    return train, test

def train_benchmark(df, random_state=42):
    numeric = [c for c in df.columns if c not in [TARGET, "crop", "region"]]
    categorical = ["crop", "region"]
    train, test = split_temporal(df)
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]
    results, fitted = [], {}
    for name, model in build_models(random_state).items():
        pipe = make_pipeline(model, numeric, categorical)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        results.append({"model": name, "MAE": mean_absolute_error(y_test, pred),
                        "RMSE": rmse, "R2": r2_score(y_test, pred)})
        fitted[name] = pipe
    metrics = pd.DataFrame(results).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    return metrics, fitted, train, test

def save_model(model, path="models/best_model.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path="models/best_model.joblib"):
    return joblib.load(path)
