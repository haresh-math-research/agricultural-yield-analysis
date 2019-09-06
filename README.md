# Agricultural Yield Intelligence & Forecasting System

A research-grade, reproducible Python platform for agricultural yield analysis, machine-learning prediction, time-series forecasting, explainable AI, uncertainty estimation, scenario analysis, and an interactive dashboard.

## What is included

- Synthetic-but-realistic agricultural dataset generator for immediate execution
- Data validation and preprocessing
- Feature engineering
- Statistical analysis and OLS regression
- Multiple ML models: Linear/Ridge/Lasso, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting
- Temporal train/test evaluation and cross-validation
- Time-series forecasting with a robust seasonal-naive baseline and optional statsmodels SARIMAX
- SHAP explainability when SHAP is installed
- Bootstrap prediction intervals
- Yield-risk classification
- Scenario simulation
- Streamlit dashboard
- FastAPI prediction API
- Automated tests
- Docker support
- CI workflow
- Research-oriented reports and configuration

## Important data note

The repository is designed to run immediately without proprietary data. The included generator creates a transparent demonstration dataset. Replace it with a documented observed dataset before making empirical research claims.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

python scripts/run_pipeline.py
streamlit run dashboard/app.py
uvicorn api.main:app --reload
```

The pipeline writes outputs to `reports/outputs/`.

## Project structure

```text
agricultural-yield-analysis/
├── api/
├── config/
├── dashboard/
├── data/
├── notebooks/
├── reports/
├── scripts/
├── src/agri_yield/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Research questions

1. How accurately can yield be predicted from climate and agronomic variables?
2. Which variables contribute most strongly to yield variability?
3. How do nonlinear ML models compare with conventional regression?
4. How does climate variability affect projected productivity?
5. Can uncertainty-aware predictions provide useful decision support?

## Disclaimer

This is an analytical and decision-support prototype. Predictions are not agronomic guarantees. Real deployment requires validated local observations, domain expertise, data governance, and independent external validation.
