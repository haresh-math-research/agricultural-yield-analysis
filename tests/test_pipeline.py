import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agri_yield.data import generate_dataset
from agri_yield.features import engineer_features
from agri_yield.models import train_benchmark

def test_dataset_generation():
    df = generate_dataset(5)
    assert len(df) > 0
    assert "yield_tonnes_per_hectare" in df.columns

def test_feature_engineering():
    df = engineer_features(generate_dataset(3))
    assert "temperature_stress" in df.columns
    assert "climate_stress_index" in df.columns

def test_model_training():
    df = engineer_features(generate_dataset(8))
    metrics, fitted, train, test = train_benchmark(df)
    assert len(metrics) >= 5
    assert not test.empty
    assert metrics["RMSE"].notna().all()
