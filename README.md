# 🌦️ Weather Forecasting with Time Series Models

> Multi-step temperature & precipitation forecasting using classical and deep learning approaches — SARIMA, Prophet, and LSTM.

## 🎯 Project Overview

This project explores **time series forecasting** applied to historical weather data, comparing statistical and neural network models to predict future temperature and precipitation values across multiple horizons (1-day, 7-day, 30-day).

**Main Goal**: Build and evaluate forecasting pipelines that can be adapted to any weather station dataset, with structured model comparison and reproducible experiments.

---

## 🏆 Models & Results

| Model | MAE (temp °C) | RMSE | Notes |
|-------|:---:|:---:|---|
| Baseline (naive) | ~3.1 | ~4.2 | Prior-day carry-forward |
| SARIMA | ~1.8 | ~2.5 | Seasonal decomposition |
| Prophet | ~1.6 | ~2.3 | Holiday & trend effects |
| LSTM (stacked) | ~1.4 | ~2.0 | Best on longer horizons |

---

## 🔍 What Was Built

### 1. Data Pipeline
- Ingestion and cleaning of raw weather station data
- Handling of missing values (interpolation + flagging)
- Feature extraction: lag features, rolling statistics, cyclical encoding of date components

### 2. Classical Models
- **SARIMA**: Grid search over (p,d,q)(P,D,Q,s) with AIC selection
- **Prophet**: Additive model with weekly/yearly seasonality and changepoint detection

### 3. Deep Learning — LSTM
- Stacked LSTM with dropout regularization
- Sliding-window sequence generation for multi-step forecasting
- Early stopping + learning rate scheduling

### 4. Evaluation Framework
- Walk-forward validation (no data leakage)
- Metrics: MAE, RMSE, MAPE
- Residual analysis and forecast visualisations

---

## 📁 Project Structure

```
weather-forecasting-timeseries/
├── data/
│   ├── raw/            # Original weather station CSVs
│   └── processed/      # Cleaned & feature-engineered datasets
├── models/             # Serialised model artefacts (.pkl / .h5)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_sarima_prophet.ipynb
│   └── 03_lstm.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models/
│   │   ├── sarima_model.py
│   │   ├── prophet_model.py
│   │   └── lstm_model.py
│   └── evaluate.py
├── reports/            # Plots and summary reports
├── requirements.txt
└── project_structure_run.py
```

---

## ⚙️ Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/pelabdang/weather-forecasting-timeseries.git
cd weather-forecasting-timeseries
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialise project structure
python project_structure_run.py

# 4. Run notebooks in order
jupyter lab notebooks/
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Prophet](https://img.shields.io/badge/Prophet-Meta-blue)
![statsmodels](https://img.shields.io/badge/statsmodels-SARIMA-green)
![Pandas](https://img.shields.io/badge/Pandas-data-lightgrey?logo=pandas)

---

## 📊 Key Takeaways

- LSTM outperforms classical models on **longer forecast horizons** (7–30 days)
- Prophet handles **trend shifts** and seasonality with minimal tuning
- Walk-forward validation is essential — standard train/test splits **overestimate** model performance on time series

---

**📧 Contact**: Angelo Pelisson · [GitHub](https://github.com/pelabdang)
