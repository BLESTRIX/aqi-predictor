# AQI Predictor 🌍💨

[![Feature Pipeline](https://github.com/user/aqi-predictor/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/user/aqi-predictor/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/user/aqi-predictor/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/user/aqi-predictor/actions/workflows/training_pipeline.yml)
[![Tests](https://github.com/user/aqi-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/user/aqi-predictor/actions/workflows/tests.yml)

An end-to-end production ML system for forecasting Air Quality Index (AQI) using live Open-Meteo weather and pollutant data, Hopsworks Feature Store, XGBoost/Scikit-Learn ML models, automated GitHub Actions CI/CD pipelines, and an interactive Streamlit dashboard.

---

## 🏛 Architecture Overview

```
                        +----------------------+
                        |   Open-Meteo API     |
                        | (Pollutants & Weather|
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        |   Feature Pipeline   |
                        | (Lag, Rolling, Time) |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        |  Hopsworks Feature   |
                        |        Store         |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        |  Training Pipeline   |
                        | (XGBoost / Registry) |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        |   Streamlit App UI   |
                        | (Forecast & SHAP)    |
                        +----------------------+
```

---

## 📁 Repository Structure

```
aqi-predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml      # CI/CD: Hourly feature extraction & sync
│       ├── training_pipeline.yml     # CI/CD: Daily batch retraining & registry promotion
│       └── tests.yml                 # CI/CD: Automated unit & integration testing
├── config/
│   ├── config.yaml                   # Feature specs, model hyperparameters, API limits
│   └── logging.yaml                  # Structured logging configuration
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis & pollutant trends
│   └── 02_shap_explainability.ipynb # SHAP/LIME feature importance experiments
├── src/                              # Main Python package
│   ├── __init__.py
│   ├── config.py                     # Pydantic Settings / YAML config loader
│   ├── data/
│   │   ├── __init__.py
│   │   ├── api_client.py             # Open-Meteo / AQICN API fetchers
│   │   └── backfill.py               # Historical raw data fetcher
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py         # Time, lag, and AQI change calculations
│   │   └── feature_store.py          # Hopsworks feature store sync interface
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                  # Scikit-learn, XGBoost, PyTorch training routines
│   │   ├── evaluate.py               # RMSE, MAE, R² metrics & SHAP evaluations
│   │   └── registry.py               # Hopsworks Model Registry saving/loading operations
│   └── utils/
│       ├── __init__.py
│       └── logger.py                 # Centralized logging setup
├── pipelines/                        # Entrypoint scripts for orchestration/automation
│   ├── run_feature_pipeline.py       # Orchestrates API -> Features -> Feature Store
│   ├── run_backfill_pipeline.py      # One-time/periodic historical backfill runner
│   └── run_training_pipeline.py      # Orchestrates Feature Store -> Train -> Evaluate -> Registry
├── app/
│   ├── streamlit_app.py              # Streamlit interactive dashboard
│   └── components/                   # UI modular components (plots, risk indicators)
├── tests/
│   ├── unit/                         # Unit tests for feature transforms and API parsers
│   └── integration/                  # Integration tests for Feature Store & Registry ops
├── .env.example                      # Template for secrets
├── Dockerfile                        # Containerization configuration
├── pyproject.toml                    # Package configuration and dependency management
├── requirements.txt                  # Frozen python requirements
└── README.md
```

---

## ⚡ Quickstart

### 1. Installation

Clone the repository and install dependencies in editable mode:

```bash
git clone https://github.com/user/aqi-predictor.git
cd aqi-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and fill in your Hopsworks API key and coordinates:

```bash
cp .env.example .env
```

### 3. Run Pipelines

**Historical Backfill:**
```bash
python pipelines/run_backfill_pipeline.py --days 30
```

**Feature Sync Pipeline (Hourly):**
```bash
python pipelines/run_feature_pipeline.py
```

**Model Retraining & Registry Pipeline (Daily):**
```bash
python pipelines/run_training_pipeline.py
```

### 4. Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Testing

Run automated tests with `pytest`:

```bash
pytest tests/
```

---

## 🐳 Docker Deployment

Build and launch via Docker:

```bash
docker build -t aqi-predictor:latest .
docker run -p 8501:8501 --env-file .env aqi-predictor:latest
```
