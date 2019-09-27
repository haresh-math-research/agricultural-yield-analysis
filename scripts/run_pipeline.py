import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agri_yield.pipeline import run

if __name__ == "__main__":
    metrics, forecast, best = run()
    print("\nModel benchmark:")
    print(metrics.to_string(index=False))
    print(f"\nSelected model: {best}")
    print("\nForecast:")
    print(forecast.to_string(index=False))
