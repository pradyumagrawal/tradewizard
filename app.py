
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date, timedelta

import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="TradeWizard", page_icon="📈", layout="wide")
# Utilities
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True
    )
    if isinstance(df.columns, pd.MultiIndex):
        # In case of multi-index columns, keep Close and Volume
        df = df[['Close', 'Volume']].copy()
    else:
        df = df[['Close', 'Volume']].copy()
    df = df.dropna().copy()
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Simple moving averages
    out['sma_5']  = out['Close'].rolling(window=5, min_periods=5).mean()
    out['sma_10'] = out['Close'].rolling(window=10, min_periods=10).mean()
    out['sma_20'] = out['Close'].rolling(window=20, min_periods=20).mean()

    # Exponential moving averages
    out['ema_12'] = out['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    out['ema_26'] = out['Close'].ewm(span=26, adjust=False, min_periods=26).mean()

    # MACD (12,26,9)
    out['macd']        = out['ema_12'] - out['ema_26']
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False, min_periods=9).mean()
    out['macd_hist']   = out['macd'] - out['macd_signal']

    # RSI(14)
    out['rsi_14'] = rsi(out['Close'], window=14)

    # Volatility proxy
    out['rolling_std_10'] = out['Close'].pct_change().rolling(10, min_periods=10).std()

    # Lags of close
    for k in range(1, 6):
        out[f'close_lag_{k}'] = out['Close'].shift(k)

    # Returns
    out['ret_1']  = out['Close'].pct_change(1)
    out['ret_5']  = out['Close'].pct_change(5)
    out['ret_10'] = out['Close'].pct_change(10)

    # Calendar/seasonality
    out['dow'] = out.index.dayofweek  # 0=Mon

    return out

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use Wilder's smoothing via ewm with alpha=1/window (adjust=False)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def make_feature_matrix(df_feat: pd.DataFrame):
    feature_cols = [
        'sma_5','sma_10','sma_20',
        'ema_12','ema_26',
        'macd','macd_signal','macd_hist',
        'rsi_14',
        'rolling_std_10',
        'close_lag_1','close_lag_2','close_lag_3','close_lag_4','close_lag_5',
        'ret_1','ret_5','ret_10',
        'Volume','dow'
    ]
    X = df_feat[feature_cols].copy()
    y = df_feat['Close'].shift(-1).copy()  # predict next-step close
    data = pd.concat([X, y.rename('target')], axis=1).dropna()
    return data.drop(columns=['target']), data['target']

def time_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    n = len(X)
    n_train = int(np.floor(n * (1 - test_size)))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    return X_train, X_test, y_train, y_test

def choose_model(name: str):
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    else:
        return LinearRegression()

def evaluate(y_true: pd.Series, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def forecast_iterative(df_base: pd.DataFrame, model, horizon: int) -> pd.DataFrame:
    """
    Iteratively forecasts next-step Close for `horizon` business days by
    appending predictions and recomputing features each step.
    """
    recent = df_base[['Close', 'Volume']].copy()
    last_volume = recent['Volume'].iloc[-1] if 'Volume' in recent else 0.0

    preds = []
    dates = []

    # Create future business day index
    last_dt = recent.index[-1]
    future_idx = pd.bdate_range(start=last_dt + pd.Timedelta(days=1), periods=horizon)

    for i, ts in enumerate(future_idx):
        # Build a temp frame including predicted closes so far
        tmp = recent.copy()
        # Recompute indicators and features
        tmp_feat = add_indicators(tmp)
        X_tmp, _ = make_feature_matrix(tmp_feat)
        if len(X_tmp) == 0:
            break
        x_last = X_tmp.iloc[[-1]]  # last row features
        yhat = float(model.predict(x_last)[0])

        preds.append(yhat)
        dates.append(ts)

        # Append for next step
        recent.loc[ts, 'Close'] = yhat
        recent.loc[ts, 'Volume'] = last_volume  # simple carry-forward

    return pd.DataFrame({'Forecast': preds}, index=pd.Index(dates, name='Date'))

def plot_results(df: pd.DataFrame, y_test: pd.Series, y_pred_test: np.ndarray, fcst: pd.DataFrame, split_idx: int):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Close'], label='Close', color='tab:blue', linewidth=1.5)

    # Backtest region
    test_index = y_test.index
    ax.plot(test_index, y_pred_test, label='Backtest Pred', color='tab:orange', linewidth=1.5)

    # Forecast region
    if fcst is not None and len(fcst) > 0:
        ax.plot(fcst.index, fcst['Forecast'], label='Forecast', color='tab:green', linestyle='--', linewidth=1.8)

    # Split marker
    if split_idx is not None:
        ax.axvline(df.index[split_idx], color='gray', linestyle=':', linewidth=1)

    ax.set_title('Historical, Backtest Prediction, and Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# ------------------------
# UI
# ------------------------
st.title("📈 TradeWizard (Your Trading Companion)")
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL")
    colA, colB = st.columns(2)
    with colA:
        start_date = st.date_input("Start date", value=date.today() - timedelta(days=365*5))
    with colB:
        end_date = st.date_input("End date", value=date.today())

    interval = st.selectbox("Interval", options=["1d", "1h", "30m", "15m"], index=0)
    model_name = st.selectbox("Model", options=["Random Forest", "Linear Regression"], index=0)
    test_size = st.slider("Test size (holdout %)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    horizon = st.slider("Forecast horizon (business days)", min_value=1, max_value=60, value=10, step=1)

    run_btn = st.button("Run")

if run_btn:
    try:
        df = fetch_prices(ticker.strip().upper(), pd.Timestamp(start_date), pd.Timestamp(end_date), interval)
        if df.empty:
            st.warning("No data returned; check ticker, dates, or interval.")
            st.stop()

        # Engineer features
        df_feat = add_indicators(df)
        X, y = make_feature_matrix(df_feat)

        if len(X) < 100:
            st.warning("Not enough rows after feature engineering; try extending the date range.")
            st.stop()

        # Time-based split
        X_train, X_test, y_train, y_test = time_split(X, y, test_size=float(test_size))
        split_idx = len(X_train)

        # Train
        model = choose_model(model_name)
        model.fit(X_train, y_train)

        # Backtest
        y_pred_test = model.predict(X_test)
        mae, rmse, r2 = evaluate(y_test, y_pred_test)

        # Forecast
        fcst = forecast_iterative(df_feat, model, horizon=horizon)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (test)", f"{mae:,.4f}")
        m2.metric("RMSE (test)", f"{rmse:,.4f}")
        m3.metric("R² (test)", f"{r2:,.4f}")

        # Plot
        plot_results(df, y_test, y_pred_test, fcst, split_idx)

        # Downloads
        col1, col2 = st.columns(2)
        with col1:
            bt_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test}, index=y_test.index)
            st.download_button(
                "Download backtest CSV",
                bt_df.to_csv(index=True).encode("utf-8"),
                file_name=f"{ticker}_backtest.csv",
                mime="text/csv",
            )
        with col2:
            if fcst is not None and len(fcst) > 0:
                st.download_button(
                    "Download forecast CSV",
                    fcst.to_csv(index=True).encode("utf-8"),
                    file_name=f"{ticker}_forecast.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.error(f"Error: {e}")

