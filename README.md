# ⚡ AI Energy Efficiency Monitoring System

> **Production-grade AI system** for building energy analytics, forecasting, anomaly detection, and real-time simulation.

---

## Project Structure

```
ai-energy-monitoring/
│
├── data/                        ← (optional) symlink / copy of raw CSVs
│
├── notebooks/
│   ├── eda.ipynb                ← Exploratory Data Analysis
│   └── feature_engineering.ipynb  ← Full ML pipeline (run this first!)
│
├── models/
│   ├── <best>_model.pkl         ← Saved best model (auto-generated)
│   └── feature_cols.pkl         ← Feature column list (auto-generated)
│
├── streamlit/
│   └── app.py                   ← Interactive Streamlit dashboard
│
├── train.csv
├── weather_train.csv
├── building_metadata.csv
├── test.csv
├── weather_test.csv
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML pipeline notebook

Open `notebooks/feature_engineering.ipynb` and run all cells.  
This will:
- Merge all data sources
- Handle missing values via **interpolation** (not ffill/bfill)
- Engineer **30+ features** including lag, rolling, cyclic time & building-type dummies
- Train **5 models**: LinearRegression · RandomForest · XGBoost · LightGBM · **LightGBM_Tuned** (Optuna)
- Run **TimeSeriesSplit 5-fold cross-validation**
- Auto-tune hyperparameters with **Optuna** (20 trials, TPE sampler)
- Compare models on **MAE · RMSE · R² · RMSLE** (4 metrics)
- Explain predictions with **SHAP** (beeswarm + importance bar)
- Detect anomalous buildings via **IsolationForest**
- Plot actual vs predicted scatter (test set)
- Save the best model + artifacts to `models/`

### 3. Launch the Streamlit dashboard

```bash
streamlit run streamlit/app.py
```

Dashboard pages:
| Page | Content |
|------|---------|
| 📊 Overview | KPIs, hourly profile, monthly trend, model leaderboard |
| 🏢 Building Analysis | Top consumers, building-type breakdown, individual time series |
| 🌤️ Weather vs Energy | Correlation analysis, heatmap, scatter + box explorer |
| 🤖 AI Forecast | 24-hour forecast, actual vs predicted scatter, MAE/RMSE/R² |
| 🚨 Anomaly Detection | IsolationForest building map, anomaly table, live IQR check |
| 🔴 Live Simulation | Real-time step-by-step prediction replay with rolling KPIs |

---

## Feature Engineering Highlights

| Feature Group | Features |
|---------------|----------|
| Time | `hour`, `day`, `month`, `weekday`, `is_weekend`, `quarter` |
| Cyclic | `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos` |
| Weather | `air_temperature`, `dew_temperature`, `cloud_coverage`, `wind_speed`, `wind_direction`, `sea_level_pressure`, `precip_depth_1_hr` |
| Building | `square_feet`, `floor_count`, `year_built`, `building_id`, `meter`, `site_id` |
| Lag | `lag_1` (1h), `lag_24` (24h), `lag_168` (1 week) |
| Rolling | `rolling_mean_24`, `rolling_mean_168`, `rolling_std_24` |
| Encoding | `use_Education`, `use_Office`, `use_Retail`, … (one-hot, 16 categories) |

---

## Model Comparison

All models are evaluated on a **chronological hold-out** (last 20% by time) — no data leakage.

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Random Forest | `n_estimators=100`, `max_depth=12` |
| XGBoost | `n_estimators=300`, `lr=0.05` |
| LightGBM | `n_estimators=300`, `lr=0.05` |
| **LightGBM_Tuned** | Optuna 20-trial HPT — best params saved to `models/best_hyperparams.pkl` |

All evaluated on a chronological 80/20 hold-out (no data leakage) with 4 metrics: **MAE · RMSE · R² · RMSLE**.

---

## Dataset

Building energy meter dataset — ~20 million hourly rows · 1,449 buildings · 16 sites · 1 year of readings.  
Includes electricity, chilled water, steam, and hot water meter types across diverse building categories.

---

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `Optuna` · `SHAP` · `statsmodels` · `Streamlit` · `Matplotlib` · `Seaborn` · `joblib`
