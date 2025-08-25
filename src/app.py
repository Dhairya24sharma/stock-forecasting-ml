# src/app.py
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import importlib
import runpy
import os
import traceback

st.set_page_config(page_title="Stock Forecasting", layout="wide")
st.title("📈 Stock Forecasting (Streamlit)")

st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Ticker (e.g., AAPL, TCS.NS):", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=dt.date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", value=dt.date.today())
horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=60, value=7, step=1)

st.sidebar.markdown("---")
use_project_trainer = st.sidebar.checkbox("Try project trainer (src/train_stock_models.py)", value=False)

@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df

def simple_lr_forecast(df, horizon):
    # Create a very simple feature: use previous close to predict next close
    data = df[["Close"]].dropna().copy()
    data["lag1"] = data["Close"].shift(1)
    data = data.dropna()

    X = data[["lag1"]].values
    y = data["Close"].values

    # Train/test split (last 20% test)
    n = len(data)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Rolling forecast for horizon using last known value iteratively
    last_close = float(df["Close"].iloc[-1])
    preds = []
    prev = last_close
    for _ in range(horizon):
        p = float(model.predict(np.array([[prev]])))
        preds.append(p)
        prev = p

    future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="B")
    forecast_df = pd.DataFrame({"PredictedClose": preds}, index=future_index)
    return mae, r2, forecast_df

def try_project_trainer(df, horizon):
    """
    Tries to call a function from src/train_stock_models.py if one exists.
    Expected signature (loosely): train_and_forecast(df, horizon) -> (metrics_dict, forecast_df)
    Falls back to running the script if needed, else raises.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "train_stock_models", os.path.join(os.path.dirname(__file__), "train_stock_models.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # noqa

        if hasattr(mod, "train_and_forecast"):
            metrics, forecast_df = mod.train_and_forecast(df, horizon)
            return metrics, forecast_df
        else:
            # As a last resort, just execute the script so any side-effects (like saved models/plots) happen.
            runpy.run_path(os.path.join(os.path.dirname(__file__), "train_stock_models.py"), run_name="__main__")
            raise AttributeError("No train_and_forecast(df, horizon) found in train_stock_models.py")

    except Exception as e:
        raise RuntimeError("Project trainer invocation failed:\n" + traceback.format_exc()) from e

if st.button("Run"):
    if not ticker.strip():
        st.error("Please enter a valid ticker.")
        st.stop()

    with st.spinner("Downloading data..."):
        df = load_data(ticker.strip(), start_date, end_date)

    if df is None or df.empty:
        st.error("No data returned. Try another ticker or date range.")
        st.stop()

    st.subheader(f"Raw Data: {ticker}")
    st.dataframe(df.tail())

    st.subheader("Price Chart")
    st.line_chart(df["Close"])

    if use_project_trainer:
        st.info("Trying project trainer from src/train_stock_models.py ...")
        try:
            metrics, forecast = try_project_trainer(df, horizon)
            st.success("Project trainer ran successfully.")
            if isinstance(metrics, dict):
                st.json(metrics)
            if isinstance(forecast, pd.DataFrame) and not forecast.empty:
                st.subheader("Forecast (Project Trainer)")
                st.line_chart(forecast)
                st.dataframe(forecast)
        except Exception as e:
            st.error(str(e))
            st.info("Falling back to simple baseline model...")
            use_project_trainer = False  # fallback to baseline

    if not use_project_trainer:
        st.subheader("Baseline Model (Lag-1 Linear Regression)")
        mae, r2, forecast_df = simple_lr_forecast(df, horizon)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE", f"{mae:,.4f}")
        with col2:
            st.metric("R²", f"{r2:,.4f}")

        st.subheader("Forecast (Next Business Days)")
        st.line_chart(forecast_df["PredictedClose"])
        st.dataframe(forecast_df)
