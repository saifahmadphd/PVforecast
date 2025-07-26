"""
PV power forecasting with XGBoost (H-hour ahead)
-------------------------------------------------
* Forecast horizon set by H (default 12 h).
* Uses CAMS (GHI) + ERA5 air temperature.
* Feature set: past PV, temp, seasonal encodings, rolling mean, etc.
* Outputs RMSE and a plot comparing observed vs forecast PV plus temperature.

Run:
    python pv_forecast_xgb_clean.py
"""

from __future__ import annotations

# === Imports =================================================================
from datetime import datetime
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error

from pvlib.iotools import get_cams
from pvlib.temperature import sapm_cell

plt.rc('text', usetex=False)

# === Config ==================================================================
HORIZON_H = 36                 # forecast horizon (hours)
LAGS = ( 18, 24,36)           # pv_power lags to include
ROLL_WIN =     2                   # rolling mean window (hours)
TRAIN_FRACTION = 0.90          # train/test split fraction
SEED = 42

# === 1) Fetch irradiance & temperature =======================================

def fetch_data(lat: float, lon: float,
               start: datetime, end: datetime,
               email: str, identifier: str) -> pd.DataFrame:
    """Download hourly GHI (CAMS) and 2 m air temperature (ERA5 via open-meteo).

    Returns a DataFrame with columns ['ghi', 'temp_air'] indexed by UTC timestamps.
    """
    ghi = get_cams(lat, lon, start=start, end=end,
                   email=email, identifier=identifier,
                   time_step='1h', map_variables=True)[0]['ghi']

    url = (
        'https://archive-api.open-meteo.com/v1/era5'
        f'?latitude={lat}&longitude={lon}'
        f'&start_date={start.date()}&end_date={end.date()}'
        '&hourly=temperature_2m'
    )
    js = requests.get(url).json()['hourly']
    times = pd.to_datetime(js['time'], utc=True)
    temp_air = pd.Series(js['temperature_2m'], index=times, name='temp_air')

    df = pd.DataFrame({'ghi': ghi}).join(temp_air, how='inner')
    return df

# === 2) Simple DC PV model ====================================================

def compute_pv_power(df: pd.DataFrame) -> pd.DataFrame:
    """Add cell temperature and PV power estimate to df.

    Uses SAPM cell temperature model and a crude efficiency correction.
    """
    a, b, deltaT = -3.47, -0.0594, 36.0
    df['cell_temp'] = sapm_cell(
        poa_global=df['ghi'], temp_air=df['temp_air'], wind_speed=0.0,
        a=a, b=b, deltaT=deltaT,
    )
    p_coeff, eff, area = -0.005, 0.18, 100.0  # example array
    df['pv_power_kW'] = (
        df['ghi'] * area * eff * (1 + p_coeff * (df['cell_temp'] - 25))
    ) * 1e-3
    return df

# === 3) Feature engineering ===================================================

def build_df_feat(df: pd.DataFrame,
                  H: int = HORIZON_H,
                  lags = LAGS,
                  roll_win: int = ROLL_WIN) -> tuple[pd.DataFrame, list[str], str]:
    """Create features/target for XGB.

    Returns (df_feat, feature_cols, target_col).
    """
    df_feat = pd.DataFrame({
        'pv_power': df['pv_power_kW'],
        'temp_air': df['temp_air'],
    })

    # Seasonal encodings
    doy = df.index.dayofyear
    df_feat['sin_doy'] = np.sin(2 * np.pi * doy / 365)
    df_feat['cos_doy'] = np.cos(2 * np.pi * doy / 365)
    df_feat['hour']    = df.index.hour

    # Rolling mean of PV
    df_feat[f'pv_roll_{roll_win}h'] = df_feat['pv_power'].rolling(roll_win, min_periods=1).mean()

    # Lags of PV
    for lag in lags:
        df_feat[f'lag_{lag}h'] = df_feat['pv_power'].shift(lag)

    # Target
    target_col = f'target_{H}h'
    df_feat[target_col] = df_feat['pv_power'].shift(-H)

    # Drop rows with NaNs in any used column
    feature_cols = [c for c in df_feat.columns if c != target_col]
    df_feat = df_feat.dropna(subset=feature_cols + [target_col])

    return df_feat, feature_cols, target_col

# === 4) Train & evaluate XGB ==================================================

def train_xgb(df_feat: pd.DataFrame, feature_cols: list[str], target_col: str,
              train_frac: float = TRAIN_FRACTION,
              seed: int = SEED):
    X = df_feat[feature_cols]
    y = df_feat[target_col]

    split = int(len(df_feat) * train_frac)
    dtrain = xgb.DMatrix(X.iloc[:split], label=y.iloc[:split])
    dtest  = xgb.DMatrix(X.iloc[split:], label=y.iloc[split:])

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': seed,
    }

    bst = xgb.train(
        params, dtrain, num_boost_round=400,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=20, verbose_eval=False,
    )

    y_pred = np.clip(bst.predict(dtest), 0, None)
    rmse = np.sqrt(mean_squared_error(y.iloc[split:], y_pred))

    return (
        y.iloc[split:],          # y_true
        y_pred,                  # y_pred
        rmse,                    # rmse
        df_feat.index[split:],   # index for plotting
        bst                      # model
    )

# === 5) Plotting ==============================================================

def plot_results(idx, y_true, y_pred, temp_series, title: str):
    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(idx, y_true, label='Observed PV', color='C0')
    ax1.plot(idx, y_pred, label='XGB Forecast', color='C1')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('PV Power (kW)')
    ax1.legend(loc='upper left', fontsize=8)

    plt.title(title)
    plt.tight_layout()
    plt.show()

# === 6) Script entry-point ====================================================

if __name__ == '__main__':
    lat, lon = 43.3026, 5.3691
    start = datetime(2023, 1, 23, 5)
    end   = datetime(2023, 12, 23, 5)

    df = fetch_data(lat, lon, start, end,
                    email='saifiitp16@gmail.com',
                    identifier='cams_radiation')
    df = compute_pv_power(df)

    df_feat, feature_cols, target_col = build_df_feat(df, H=HORIZON_H)

    y_true, y_pred, rmse, idx, model = train_xgb(df_feat, feature_cols, target_col)

    print(f"{HORIZON_H}-h-ahead RMSE (non-neg): {rmse:.3f} kW")

    temp_plot = df_feat.loc[idx, 'temp_air']
    plot_results(idx, y_true, y_pred, temp_plot,
                 title=f"PV Forecast vs Temperature ({HORIZON_H} h ahead)")
