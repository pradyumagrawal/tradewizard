import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date, timedelta

import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import traceback

# Remove the problematic cache clearing code
# if 'st' in globals():
#     st.cache_data.clear()

# Instead, initialize session state if needed
if 'init' not in st.session_state:
    st.session_state.init = True
    try:
        st.cache_data.clear()
    except:
        pass

st.set_page_config(page_title="TradeWizard", page_icon="📈", layout="wide")
# Utilities
@st.cache_data(ttl=3600)
def fetch_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True
        )
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        if isinstance(df.columns, pd.MultiIndex):
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        else:
            # Keep all available columns
            keep_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns]
            df = df[keep_cols].copy()
        return df.dropna().copy()
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Simple moving averages
    out['sma_5' ] = out['Close'].rolling(window=5, min_periods=5).mean()
    out['sma_10'] = out['Close'].rolling(window=10, min_periods=10).mean()
    out['sma_20'] = out['Close'].rolling(window=20, min_periods=20).mean()
    out['sma_50'] = out['Close'].rolling(window=50, min_periods=20).mean()

    # Exponential moving averages
    out['ema_12'] = out['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    out['ema_26'] = out['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    out['ema_50'] = out['Close'].ewm(span=50, adjust=False, min_periods=50).mean()
    out['ema_100'] = out['Close'].ewm(span=100, adjust=False, min_periods=100).mean()

    # MACD (12,26,9)
    out['macd'       ] = out['ema_12'] - out['ema_26']
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False, min_periods=9).mean()
    out['macd_hist'  ] = out['macd'] - out['macd_signal']

    # RSI(14)
    out['rsi_14'] = rsi(out['Close'], window=14)

    # Volatility proxy
    out['rolling_std_10'] = out['Close'].pct_change().rolling(10, min_periods=10).std()
    out['rolling_std_20'] = out['Close'].pct_change().rolling(20, min_periods=10).std()

    # Rolling mean momentum
    out['roll_mean_20'] = out['Close'].rolling(20, min_periods=10).mean()
    out['roll_mean_50'] = out['Close'].rolling(50, min_periods=20).mean()

    # Lags of close
    for k in range(1, 6):
        out[f'close_lag_{k}'] = out['Close'].shift(k)

    # Returns
    out['ret_1' ] = out['Close'].pct_change(1)
    out['ret_5' ] = out['Close'].pct_change(5)
    out['ret_10'] = out['Close'].pct_change(10)

    # Price trend features
    out['price_momentum'] = out['Close'].pct_change(5)
    out['price_acceleration'] = out['price_momentum'].diff(1)
    
    # Volatility features
    out['volatility'] = out['Close'].pct_change().rolling(window=20).std()
    out['high_vol'] = (out['volatility'] > out['volatility'].rolling(window=50).mean()).astype(int)
    
    # Add range-based features only if High/Low available
    if 'High' in out.columns and 'Low' in out.columns:
        out['daily_range'] = (out['High'] - out['Low']) / out['Close']
        out['range_ma'] = out['daily_range'].rolling(window=10).mean()

    # Calendar/seasonality
    out['dow'] = out.index.dayofweek  # 0=Mon
    out['dayofmonth'] = out.index.day
    out['dayofyear'] = out.index.dayofyear
    out['weekofyear'] = out.index.isocalendar().week.astype(int)
    out['month'] = out.index.month
    out['is_month_end'] = out.index.is_month_end.astype(int)
    
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
        'sma_5','sma_10','sma_20','sma_50',
        'ema_12','ema_26','ema_50','ema_100',
        'macd','macd_signal','macd_hist',
        'rsi_14',
        'rolling_std_10','rolling_std_20',
        'roll_mean_20','roll_mean_50',
        'close_lag_1','close_lag_2','close_lag_3','close_lag_4','close_lag_5',
        'ret_1','ret_5','ret_10',
        'Volume','dow','dayofmonth','dayofyear','weekofyear','month','is_month_end'
    ]
    # Keep only features that exist in the DataFrame
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    X = df_feat[feature_cols].copy()
    y = df_feat['Close'].shift(-1).copy()  # predict next-step close

    # Ensure target column exists by assigning into a DataFrame explicitly
    data = X.copy()
    data['target'] = y
    data = data.dropna()

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
            n_estimators=200,
            max_depth=8,  # Set specific depth to capture more patterns
            min_samples_leaf=5,  # Allow smaller leaf sizes
            random_state=42,
            n_jobs=-1
        )
    elif name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=3,
            subsample=0.8,  # Add randomness
            random_state=42
        )
    else:
        return LinearRegression()

def evaluate(y_true: pd.Series, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def forecast_iterative(df_base: pd.DataFrame, model, horizon: int) -> pd.DataFrame:
    """
    Iteratively forecasts next-step Close for `horizon` business days by
    appending predictions and recomputing features each step.
    """
    if horizon < 1:
        return pd.DataFrame()
        
    recent = df_base[['Close', 'Volume']].copy()
    if recent.empty:
        return pd.DataFrame()
        
    last_volume = recent['Volume'].iloc[-1]
    preds = []
    dates = []

    last_dt = recent.index[-1]
    future_idx = pd.bdate_range(start=last_dt + pd.Timedelta(days=1), periods=horizon)

    for ts in future_idx:
        try:
            tmp = recent.copy()
            tmp_feat = add_indicators(tmp)
            X_tmp, _ = make_feature_matrix(tmp_feat)
            
            if len(X_tmp) == 0:
                break
                
            x_last = X_tmp.iloc[[-1]]
            yhat = float(model.predict(x_last)[0])
            
            if not np.isfinite(yhat):
                break
                
            preds.append(yhat)
            dates.append(ts)

            recent.loc[ts, 'Close'] = yhat
            recent.loc[ts, 'Volume'] = last_volume
        except Exception:
            break

    if not preds:
        return pd.DataFrame()
        
    return pd.DataFrame({'Forecast': preds}, index=pd.Index(dates, name='Date'))

def forecast_iterative_adaptive(df_base: pd.DataFrame, model_template, X_history: pd.DataFrame, horizon: int, window: int = 252, retrain: bool = True, retrain_every: int = 3) -> pd.DataFrame:
    """
    Faster adaptive iterative forecast:
      - If retrain=False: reuse given trained model and a scaler fit on X_history; update only lag/return/calendar features incrementally.
      - If retrain=True: retrain on a rolling `window` but only every `retrain_every` steps to save time; indicator recomputation limited to the rolling window.
    X_history: typically the training feature matrix (X_train) used to fit the model outside this function.
    """
    if horizon < 1:
        return pd.DataFrame()

    recent = df_base[['Close', 'Volume']].copy()
    if recent.empty:
        return pd.DataFrame()

    last_volume = recent['Volume'].iloc[-1]
    preds = []
    dates = []

    last_dt = recent.index[-1]
    future_idx = pd.bdate_range(start=last_dt + pd.Timedelta(days=1), periods=horizon)

    # Precompute full features once (used for base feature set)
    try:
        X_all, _ = make_feature_matrix(add_indicators(df_base))
    except Exception:
        X_all = pd.DataFrame()
    if X_all.empty and retrain is False:
        return pd.DataFrame()

    # Setup for non-retrain fast path
    if not retrain:
        scaler = StandardScaler()
        try:
            scaler.fit(X_history)  # fit once
        except Exception:
            return pd.DataFrame()
        mdl = model_template  # assume already trained outside
        # current_close tracks the latest observed/predicted close
        current_close = recent['Close'].iloc[-1]
        # cache last row as dict for fast updates
        if not X_all.empty:
            last_row_vals = X_all.iloc[-1].to_dict()
        else:
            last_row_vals = {}

        for ts in future_idx:
            # build feature vector quickly by shifting lag features
            new_vals = last_row_vals.copy()
            # update close lags: close_lag_1 is previous close (current_close)
            if 'close_lag_1' in X_all.columns:
                new_vals['close_lag_5'] = last_row_vals.get('close_lag_4', last_row_vals.get('close_lag_5', np.nan))
                new_vals['close_lag_4'] = last_row_vals.get('close_lag_3', last_row_vals.get('close_lag_4', np.nan))
                new_vals['close_lag_3'] = last_row_vals.get('close_lag_2', last_row_vals.get('close_lag_3', np.nan))
                new_vals['close_lag_2'] = last_row_vals.get('close_lag_1', last_row_vals.get('close_lag_2', np.nan))
                new_vals['close_lag_1'] = current_close

            # update returns if present
            if 'ret_1' in X_all.columns:
                # ret_1 for the new row = (predicted / current_close) - 1 -> placeholder using current_close until we predict
                new_vals['ret_1'] = 0.0  # will be updated after prediction
            if 'ret_5' in X_all.columns:
                new_vals['ret_5'] = last_row_vals.get('ret_4', last_row_vals.get('ret_5', 0.0)) if 'ret_4' in last_row_vals else last_row_vals.get('ret_5', 0.0)
            if 'ret_10' in X_all.columns:
                new_vals['ret_10'] = last_row_vals.get('ret_10', 0.0)

            # update calendar features
            if 'dow' in X_all.columns:
                new_vals['dow'] = ts.dayofweek
            if 'dayofmonth' in X_all.columns:
                new_vals['dayofmonth'] = ts.day
            if 'dayofyear' in X_all.columns:
                new_vals['dayofyear'] = ts.timetuple().tm_yday
            if 'weekofyear' in X_all.columns:
                new_vals['weekofyear'] = int(ts.isocalendar()[1])
            if 'month' in X_all.columns:
                new_vals['month'] = ts.month
            if 'is_month_end' in X_all.columns:
                new_vals['is_month_end'] = int(ts.is_month_end)

            x_vec = pd.DataFrame([new_vals], columns=X_all.columns).fillna(method='ffill', axis=1).fillna(0.0)

            try:
                x_scaled = scaler.transform(x_vec)
                yhat = float(mdl.predict(x_scaled)[0])
                
                # Add small random variations based on historical volatility
                random_factor = np.random.normal(0, hist_volatility * 0.5)
                yhat = base_pred * (1 + random_factor)
            except Exception:
                break

            # update ret_1 properly now
            if 'ret_1' in x_vec.columns:
                x_vec.at[0, 'ret_1'] = (yhat / current_close) - 1.0

            preds.append(yhat)
            dates.append(ts)

            # update trackers for next iteration
            current_close = yhat
            last_row_vals = x_vec.iloc[0].to_dict()
            last_row_vals['close_lag_1'] = current_close

        if not preds:
            return pd.DataFrame()
        return pd.DataFrame({'Forecast': preds}, index=pd.Index(dates, name='Date'))

    # Retrain path (faster: retrain only every retrain_every steps and compute indicators on rolling window)
    mdl = None
    scaler = None
    step = 0
    for ts in future_idx:
        step += 1
        # recompute indicators on a small rolling slice (window + a few extra)
        tail_rows = int(min(len(recent), max(window, 60)))
        tmp = recent.tail(tail_rows).copy()
        tmp_feat = add_indicators(tmp)
        X_tmp, y_tmp = make_feature_matrix(tmp_feat)
        if len(X_tmp) < 10:
            break

        # choose recent training slice (within window)
        X_train_slice = X_tmp.copy()
        y_train_slice = y_tmp.copy()
        if len(X_train_slice) > window:
            X_train_slice = X_train_slice.iloc[-window:]
            y_train_slice = y_train_slice.iloc[-window:]

        # retrain only every `retrain_every` steps (and always retrain at step 1)
        if (mdl is None) or (step % max(1, retrain_every) == 1):
            scaler = StandardScaler()
            try:
                X_train_scaled = scaler.fit_transform(X_train_slice)
            except Exception:
                break
            mdl = clone(model_template)
            try:
                mdl.fit(X_train_scaled, y_train_slice)
            except Exception:
                break

        # predict using the last computed features
        x_last = X_tmp.iloc[[-1]].copy()
        try:
            x_last_scaled = scaler.transform(x_last)
            yhat = float(mdl.predict(x_last_scaled)[0])
        except Exception:
            break

        preds.append(yhat)
        dates.append(ts)

        # append predicted point so next iteration uses updated lags/indicators
        recent.loc[ts, 'Close'] = yhat
        recent.loc[ts, 'Volume'] = last_volume

    if not preds:
        return pd.DataFrame()

    return pd.DataFrame({'Forecast': preds}, index=pd.Index(dates, name='Date'))

def plot_results(df: pd.DataFrame, y_test: pd.Series, y_pred_test: np.ndarray, fcst: pd.DataFrame, split_idx: int):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Close'], label='Close', color='tab:blue', linewidth=1.5)

    # Backtest region
    test_index = y_test.index
    ax.plot(test_index, y_pred_test, label='Backtest Pred', color='tab:orange', linewidth=1.5)

    # Forecast region with confidence intervals
    if fcst is not None and len(fcst) > 0:
        # Calculate historical volatility from recent data
        recent_returns = df['Close'].pct_change().tail(20)
        hist_vol = float(recent_returns.std())  # Convert to scalar
        base_price = float(df['Close'].iloc[-1])  # Convert to scalar
        
        # Calculate confidence intervals for each forecast step
        n_steps = len(fcst)
        conf_multiplier = np.sqrt(np.arange(1, n_steps + 1))
        conf_width = hist_vol * base_price * conf_multiplier
        
        # Ensure arrays are properly aligned
        upper = fcst['Forecast'].values + conf_width
        lower = fcst['Forecast'].values - conf_width
        
        # Plot forecast and confidence interval
        ax.fill_between(fcst.index, lower, upper, color='tab:green', alpha=0.1, label='95% Confidence')
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
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    if not ticker:
        st.error("Please enter a valid ticker symbol")
        st.stop()
    colA, colB = st.columns(2)
    with colA:
        start_date = st.date_input("Start date", value=date.today() - timedelta(days=365*5))
    with colB:
        end_date = st.date_input("End date", value=date.today())

    if start_date >= end_date:
        st.error("Start date must be before end date")
        st.stop()

    interval = st.selectbox("Interval", options=["1d", "1h", "30m", "15m"], index=0)
    model_name = st.selectbox("Model", options=["Random Forest", "Gradient Boosting", "Linear Regression"], index=0)
    test_size = st.slider("Test size (holdout %)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    horizon = st.slider("Forecast horizon (business days)", min_value=1, max_value=60, value=10, step=1)

    # adaptive forecasting settings
    retrain = st.checkbox("Adaptive retrain during forecasting (recommended)", value=True)
    window_size = st.slider("Retrain window (days)", min_value=50, max_value=2000, value=252, step=1)

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
        if len(X_train) < 50:  # Minimum required samples
            st.error("Not enough training data. Please extend the date range.")
            st.stop()
        
        model = choose_model(model_name)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            st.stop()

        # Backtest
        y_pred_test = model.predict(X_test)
        mae, rmse, r2 = evaluate(y_test, y_pred_test)

        # Forecast + plotting wrapped in a spinner so users see a loading icon
        with st.spinner("Generating forecast and graph..."):
            fcst = forecast_iterative_adaptive(df_feat, model, X_train, horizon=horizon, window=int(window_size), retrain=bool(retrain), retrain_every=3)

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
        st.error("An error occurred — full traceback below:")
        st.exception(e)
        st.text(traceback.format_exc())

