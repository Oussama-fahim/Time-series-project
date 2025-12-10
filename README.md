📘 README – Deep Time Series Forecasting Project
🧠 Overview

This project focuses on building advanced forecasting models using classical statistical methods and deep learning architectures.
It includes all stages of a modern time series pipeline—from data exploration to feature engineering, modeling, evaluation, and future forecasting.

Two main workflows are included:

✔️ Deep_Time_Series.ipynb –

A full deep learning pipeline for generic time series forecasting using architectures such as LSTM, GRU, CNN-1D, and hybrid models.

✔️ M5_Forecasting.ipynb –

A complete solution to the M5 Walmart Sales Forecasting challenge, combining hierarchical forecasting, exogenous features, and deep learning models.

📂 Project Structure
📁 project/
│── Deep_Time_Series.ipynb
│── M5_Forecasting.ipynb
│── README.md
│── data/
│     ├── raw/          # Original datasets
│     ├── processed/    # Cleaned + engineered datasets


🎯 Objectives

Build robust forecasting systems for multivariate or hierarchical time series.

Compare classical and deep learning approaches:

ARIMA, SARIMA, ETS, Prophet

LSTM, GRU, Encoder–Decoder, CNN-1D, N-BEATS, Transformers

Engineer features improving predictive power:

Lags, rolling windows, seasonal indicators

Fourier transformations

Calendar & event features (M5)

Optimize models using:

Bayesian Optimization

Grid / Random Search

Evaluate with industry-standard metrics:

RMSE, MAE, MAPE, sMAPE, WRMSSE (for M5)

📊 Dataset Description
🗂️ 1. Generic Time Series Dataset

Used for exploring deep architectures:

value : observed measurements

timestamp : temporal index

Optional covariates depending on experiment

🛒 2. M5 Forecasting Dataset

Provided by Walmart:

30,490 hierarchical products

1,941 days of sales

Exogenous variables:

Prices

Calendar events

Snap events

State-level differences

🔧 Methodology
1. Data Preprocessing

Handling missing timestamps

Filling NA with domain-aware imputation

Normalization: MinMaxScaler / StandardScaler

Outlier treatment via IQR or z-score

Stationarity check:

ADF Test

KPSS Test

Optional Box-Cox transformations

2. Feature Engineering

Includes:

Lag features: t-1, t-7, t-28

Rolling statistics: mean, std, min, max

Time-based features:

dayofweek, month, year, holiday flags

Fourier series (for seasonalities > 365)

M5-specific features: price volatility, promotions

3. Modeling
🧮 Classical Models
Model	Use Case
ARIMA / SARIMA	Stationary or seasonal data
Exponential Smoothing	Trend & seasonal patterns
Prophet	Business time series
🧠 Deep Learning Models
Model	Description
LSTM	Captures long temporal dependencies
GRU	Faster and efficient variant of LSTM
Conv1D	Extracts local temporal features
Encoder–Decoder	Sequence-to-Sequence forecasting
N-BEATS	Advanced deep architecture for univariate forecasting
Transformers	SOTA method for long-range dependencies
4. Evaluation

Metrics depend on the dataset:

Generic Time Series

MAE

RMSE

MAPE

sMAPE

M5 Challenge

WRMSSE (Weighted Root Mean Squared Scaled Error)

All metrics are visualized with:

Error tables

Prediction vs actual plots

Residual analysis charts

🚀 How to Run the Project
Prerequisites

Python 3.9+

Recommended hardware: GPU for deep learning models

Install dependencies
pip install -r requirements.txt

Run the notebooks

Launch Jupyter:

jupyter notebook


Open:

Deep_Time_Series.ipynb

M5_Forecasting.ipynb

🧪 Results Summary
📈 Deep Learning Findings

LSTM and GRU models provide strong results for medium-term forecasting.

CNN-1D improves performance for noisy short-memory series.

Transformers excel in long-sequence forecasting.

🛍️ M5 Forecasting Results

Feature engineering dramatically boosts accuracy.

The best-performing model combines:

Deep learning encoder–decoder

Hierarchical reconciliation

Calendar + price features

📌 Key Strengths of the Project

✔ Extremely well-structured forecasting pipeline
✔ Professional-grade exploration and modeling
✔ Full integration of statistical and deep learning techniques
✔ Detailed reproducible workflow
✔ Ready for deployment (exported models + scripts)

🛠️ Future Improvements

Add AutoML for time series (Kats, Darts)

Serve models using FastAPI or Streamlit

Add multi-step probabilistic forecasting with:

DeepAR

Temporal Fusion Transformers (TFT)

Use mlflow for experiment tracking

👤 Author

Oussama Fahim
ENSAM – Moulay Ismail University
Deep Learning & Time Series Modeling
