# 🌤️ Islamabad AQI Predictor

A serverless, end-to-end machine learning system that forecasts Islamabad's
Air Quality Index (AQI) 24, 48, and 72 hours ahead — with hazard alerting
and SHAP-based explainability, served through a live Streamlit dashboard.

**Pipeline:** AQICN (Islamabad US Embassy station) → Hopsworks Feature Store
→ Model Registry → Streamlit Dashboard, automated end-to-end with GitHub
Actions.

---

## Why AQICN, not a modeled weather API

This project originally trained on Open-Meteo's historical air-quality API.
A diagnostic check (`src/features/check_feature_bias.py`) surfaced a
critical problem: across ~3 years of hourly data, the maximum recorded AQI
was **188.3**, and **zero** rows were ever classified "Very Unhealthy or
worse" (AQI > 200) — despite Islamabad being internationally documented as
regularly reaching hazardous winter smog levels. The hazard-alert banner
this project needed to build could **structurally never fire** on that
data, no matter how good the model was.

Historical data was re-sourced from AQICN's Islamabad US Embassy
monitoring station — the same government-grade sensor used for the live
feed — and the full pipeline was migrated and retrained. The result:

| Metric | v1: Open-Meteo (hourly) | v2: AQICN Embassy (daily) |
|---|---|---|
| Date range | 2022-08 → 2026-09 | 2019-04 → 2026-02 |
| Max recorded AQI | 188.3 | 298.0 |
| Days "Very Unhealthy" (>200) | 0 (0.00%) | 99 (3.98%) |
| 72h forecast R² | ~0.48 (estimated) | 0.750 (measured) |

The old v1 feature group/model was intentionally **kept, not deleted**, as
a documented before/after baseline. See the full `AQI_Predictor_Final_Report.docx`
for the complete writeup, including the EDA that surfaced this.

---

## Architecture

```
┌─────────────────┐    hourly     ┌──────────────────────────┐
│  AQICN API       │──────────────▶  Feature Pipeline          │
│  (live station)  │              │  src/data/api_client.py    │
└─────────────────┘              │  → build_features.py       │
                                  │  → feature_store.py        │
                                  └─────────────┬───────────────┘
                                                ▼
                                  ┌──────────────────────────┐
                                  │  Hopsworks Feature Store   │
                                  │  (v2, daily cadence)       │
                                  └─────────────┬───────────────┘
                                     daily       ▼
                                  ┌──────────────────────────┐
                                  │  Training Pipeline          │
                                  │  src/models/train.py        │
                                  │  Ridge / RF / XGBoost / MLP  │
                                  │  vs. Persistence baseline    │
                                  └─────────────┬───────────────┘
                                                ▼
                                  ┌──────────────────────────┐
                                  │  Hopsworks Model Registry   │
                                  └─────────────┬───────────────┘
                                  on-demand      ▼
                                  ┌──────────────────────────┐
                                  │  Streamlit Dashboard        │
                                  │  app/main.py                │
                                  │  forecasts + SHAP +          │
                                  │  live AQICN reading          │
                                  └──────────────────────────┘
```

| Component | Files | Schedule |
|---|---|---|
| Feature pipeline | `src/data/api_client.py` → `src/features/build_features.py` → `src/features/feature_store.py` | Hourly (GitHub Actions) |
| Backfill | `src/data/backfill.py` | One-off / on data source change |
| Training pipeline | `src/models/train.py` → `src/models/registry.py` | Daily, 02:00 UTC (GitHub Actions) |
| Inference | `src/models/predict.py` (feature vector → model → SHAP) | On dashboard load |
| Dashboard | `app/main.py` + `app/components/` | Real-time, on request |
| Feature Store / Model Registry | Hopsworks (`pakistan_aqi_predictor`) | Persistent |

---

## Model comparison

Three model families were trained and compared on an identical
chronological 80/20 train/test split (see `src/models/train.py` and
`notebooks/AQI_Model_Comparison.ipynb`):

| Model | 24h R² | 48h R² | 72h R² |
|---|---|---|---|
| Persistence (naive baseline) | 0.854 | 0.757 | 0.686 |
| **Ridge Regression (selected)** | **0.868** | **0.795** | **0.750** |
| Random Forest | 0.845 | 0.763 | 0.712 |
| XGBoost | 0.856 | 0.748 | 0.703 |
| MLP (neural net) | 0.863 | 0.778 | 0.722 |
| SARIMA | 0.782 | 0.733 | 0.690 |

**Ridge — the simplest candidate — won at every horizon.** This lines up
with the EDA: AQI shows strong short-lag autocorrelation and multi-day
persistence, which favors a model that linearly weights recent lag/rolling
features over one built for complex non-linear interactions that aren't
strongly present in this signal.

SARIMA was evaluated for comparison but is **not** a production candidate:
it requires walk-forward sequential fitting against the full time series,
which doesn't fit this repo's single-row `model.predict(feature_vector)`
serving contract in `predict.py`.

### An important caveat the metrics table doesn't show

Restricting error to only the days that matter for hazard alerting —
genuine "Unhealthy or worse" days (AQI > 150) — flips the ranking:

| Model | Tail MAE, 24h | Tail MAE, 48h | Tail MAE, 72h |
|---|---|---|---|
| **Persistence** | **12.54** | **17.68** | **19.61** |
| Ridge (production) | 14.51 | 20.41 | 23.26 |
| MLP | 14.89 | 20.92 | 22.86 |
| Random Forest | 16.04 | 22.45 | 24.89 |
| SARIMA | 17.53 | 19.62 | 22.48 |
| XGBoost | 15.58 | 23.85 | 25.96 |

**On the days a hazard-alert system exists to get right, naive persistence
("tomorrow looks like today") beats every trained model, including the
one currently deployed.** The trained models still add real value overall
(better full-dataset R² on typical days), but overall R² is the wrong
primary metric for this project's actual purpose. None of the current
models genuinely anticipate spike onset — they react to trend. See
Section 9 of the final report for the full analysis, including a
peak-lag cross-correlation showing this is an under-anticipation problem,
not a pure delay.

---

## Explainability (SHAP)

Per-horizon SHAP values are computed against a background sample of 100
recent feature rows and surfaced in the dashboard as a "Why This Forecast"
panel. Across all three horizons, the dominant contributors are
consistently the current/lagged AQI value, the 7-day rolling mean, and
(at longer horizons) calendar month — consistent with the EDA's
autocorrelation and seasonality findings.

---

## Repository structure

```
├── app/
│   ├── main.py                    # Streamlit dashboard
│   └── components/
│       ├── plots.py                # gauge / trend / pollutant charts
│       └── risk_indicators.py      # single source of truth for EPA AQI colors
├── src/
│   ├── data/
│   │   ├── api_client.py           # live AQICN fetch + parsing
│   │   ├── backfill.py             # historical CSV → feature store
│   │   └── fetch_historical.py     # legacy Open-Meteo fetch, v1 baseline
│   ├── features/
│   │   ├── build_features.py       # daily cadence, lags, rolling stats, targets
│   │   ├── feature_store.py        # Hopsworks feature group push
│   │   └── check_feature_bias.py   # the diagnostic that surfaced the v1 data gap
│   ├── models/
│   │   ├── train.py                # trains + compares Ridge/RF/XGBoost/MLP
│   │   ├── evaluate.py             # metrics + SHAP explanation logic
│   │   ├── predict.py              # inference: feature vector → forecast → SHAP
│   │   └── registry.py             # Hopsworks Model Registry wrapper
│   ├── config.py
│   └── utils/logger.py
├── notebooks/
│   ├── AQI_EDA.ipynb                # exploratory data analysis
│   └── AQI_Model_Comparison.ipynb   # Persistence/Ridge/RF/XGBoost/MLP/SARIMA comparison
├── config/
│   ├── config.yaml                  # location, feature store, model config
│   └── logging.yaml
├── tests/
│   ├── unit/                        # feature engineering, API client
│   └── integration/                 # feature store, model registry (mocked Hopsworks)
├── .github/workflows/
│   ├── feature_pipeline.yml         # hourly
│   ├── training_pipeline.yml        # daily
│   └── tests.yml                    # on push/PR
├── AQI_Predictor_Final_Report.docx  # full write-up, EDA, and findings
└── requirements.txt
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd aqi-predictor
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```bash
HOPSWORKS_API_KEY=your_hopsworks_api_key
AQICN_API_KEY=your_aqicn_api_key
```

- `HOPSWORKS_API_KEY` — required for the feature store, model registry,
  training, and inference. Get one from [Hopsworks](https://www.hopsworks.ai/).
- `AQICN_API_KEY` — required for live fetches (hourly feature pipeline,
  and the dashboard's live station reading + pollutant breakdown). Get one
  from [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/).

Project-level config (location, feature group name/version, target
columns) lives in `config/config.yaml`.

### 3. Run the pipelines

```bash
# One-off: backfill historical data into the feature store
python -m src.data.backfill

# Feature pipeline (normally runs hourly via GitHub Actions)
python -m src.features.feature_store

# Training pipeline (normally runs daily via GitHub Actions)
python -m src.models.train

# Dashboard
streamlit run app/main.py
```

### 4. Run tests

```bash
pytest --cov=src tests/
```

---

## Dashboard features

- **Live station reading** — fetched directly from AQICN on page load,
  independent of the (up to hourly-stale) feature store snapshot used for
  model inputs, with a visible indicator of which source is showing.
- AQI gauge and EPA-standard health-status card for the current reading.
- 72-hour forecast cards and trend chart with shaded AQI-category bands.
- Live pollutant breakdown (PM2.5, PM10, NO₂, SO₂, O₃, CO) from the direct
  AQICN reading.
- Hazardous-AQI alert banner (triggers at AQI ≥ 200) — now meaningfully
  testable against real data, unlike under the original Open-Meteo dataset.
- SHAP-based "Why This Forecast" panel, tabbed by horizon.
- Detailed forecast breakdown table with per-horizon model confidence.

---

## CI/CD

| Workflow | File | Trigger |
|---|---|---|
| Feature pipeline | `.github/workflows/feature_pipeline.yml` | Every hour + manual dispatch |
| Training pipeline | `.github/workflows/training_pipeline.yml` | Daily, 02:00 UTC + manual dispatch |
| Test suite | `.github/workflows/tests.yml` | On push/PR to `main`/`master`/`develop` |

---

## Known limitations & priority next steps

This project's report is deliberately honest about what's unfinished
rather than presenting it as more complete than it is:

- **~27.6% of the test set is fabricated by interpolation.** A genuine
  115-day gap in the source data (2025-03-05 → 2025-06-27), among smaller
  gaps, gets linearly interpolated by `enforce_daily_cadence()`. This
  inflates every reported R²/RMSE figure — real, sensor-backed accuracy is
  closer to **0.82 / 0.73 / 0.68** (24h/48h/72h) than the headline numbers
  above. **Priority fix:** carry an `is_interpolated` flag through
  `build_features.py` and exclude those rows from evaluation (and ideally
  training).
- **No trained model beats naive persistence on hazardous-tier days**
  (see the tail-MAE table above). Future work should optimize directly for
  tail-focused error or spike-onset detection, and explore features that
  might actually signal an impending spike (wind direction/speed,
  precipitation) — none of which exist in the current PM2.5-only dataset.
- **Single station, single pollutant.** The historical record covers PM2.5
  only, from one station. Multi-pollutant history and a second station for
  cross-validation would strengthen robustness.
- **No "Hazardous" (>300) examples exist in the historical record**
  (max observed: 298) — model behavior above that threshold is untested
  extrapolation.
- Deep learning sequence models (LSTM / Temporal Fusion Transformer) have
  not yet been evaluated against Ridge/RF/XGBoost/MLP.

---

## License

Add a license (e.g. MIT) if you intend this repo to be reused — none is
currently declared in `pyproject.toml`.
