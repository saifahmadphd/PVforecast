from pvlib.iotools import get_cams
from datetime import datetime
import requests
import pandas as pd
import numpy as np
from pvlib.temperature import sapm_cell
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rc('text', usetex=False)
"""
3‑hour‑ahead PV‑power forecast with **simple XGBoost**
-----------------------------------------------------
* Forecast horizon: 3 h (suitable for very‑short‑term dispatch decisions).
* All predictions are clipped **≥ 0** so PV can never be negative.
"""

# 1) Fetch irradiance & temperature -----------------------------------------

def fetch_data(lat, lon, start, end, email, identifier):
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
    return pd.DataFrame({'ghi': ghi}).join(temp_air, how='inner')

# 2) DC PV power -------------------------------------------------------------

def compute_pv_power(df):
    a, b, deltaT = -3.47, -0.0594, 36.0
    df['cell_temp'] = sapm_cell(
        poa_global=df['ghi'], temp_air=df['temp_air'], wind_speed=0.0,
        a=a, b=b, deltaT=deltaT,
    )
    p_coeff, eff, area = -0.005, 0.18, 100.0  # 100 m² reference array
    df['pv_power_kW'] = (
        df['ghi'] * area * eff * (1 + p_coeff * (df['cell_temp'] - 25))
    ) * 1e-3
    return df

# 3) Feature engineering -----------------------------------------------------

def prepare_features(df):
    df_feat = pd.DataFrame({
        'pv_power': df['pv_power_kW'],
        'temp_air': df['temp_air'],
    })
    doy = df.index.dayofyear
    H = 12      # Prediction horizon
    df_feat['sin_doy'] = np.sin(2*np.pi*doy/365)
    df_feat['cos_doy'] = np.cos(2*np.pi*doy/365)
    df_feat['hour']      = df_feat.index.hour
    df_feat['pv_roll_2h']= df_feat['pv_power'].rolling(2, min_periods=1).mean()
    for lag in [1, 2, 3, 24]:
        df_feat[f'lag_{lag}h'] = df_feat['pv_power'].shift(lag)
    # 3‑h‑ahead target
    df_feat['target'] = df_feat['pv_power'].shift(-H)
    return df_feat.dropna()

# 4) Train & evaluate XGB ----------------------------------------------------

def train_forecast(df_feat):
    feature_cols = [
        'pv_power', 'temp_air', 'sin_doy', 'cos_doy', 'hour', 'pv_roll_2h'
    ] + [f'lag_{lag}h' for lag in [1, 2, 3, 24]]

    X = df_feat[feature_cols]
    y = df_feat['target']

    split = int(len(df_feat) * 0.99)
    dtrain = xgb.DMatrix(X.iloc[:split], label=y.iloc[:split])
    dtest  = xgb.DMatrix(X.iloc[split:], label=y.iloc[split:])

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42,
    }

    bst = xgb.train(
        params, dtrain, num_boost_round=200,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=10, verbose_eval=False,
    )

    y_pred = bst.predict(dtest)
    # ensure non‑negative PV forecast
    y_pred = np.clip(y_pred, 0, None)

    rmse = np.sqrt(mean_squared_error(y.iloc[split:], y_pred))
    print(f"3‑h‑ahead RMSE (non‑neg): {rmse:.3f} kW")

    # ---------------- Plot PV + temperature ----------------
    temp = df_feat['temp_air'].iloc[split:]      # current-hour temp
    idx  = df_feat.index[split:]

    fig, ax1 = plt.subplots(figsize=(10,4))

    # PV power (left y‑axis)
    ax1.plot(idx, y.iloc[split:],    label='Observed PV',   color='C0')
    ax1.plot(idx, y_pred,            label='XGB Forecast',  color='C1')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('PV Power (kW)')
    ax1.legend(loc='upper left', fontsize=8)

    # Temperature (right y‑axis)
    ax2 = ax1.twinx()
    ax2.plot(idx, temp,              label='Air Temp (°C)', color='C2', linestyle=':')
    ax2.set_ylabel('Air Temperature (°C)')
    ax2.legend(loc='upper right', fontsize=8)

    plt.title("PV Forecast vs Temperature")
    plt.tight_layout()
    plt.show()

# 5) Script entry‑point ------------------------------------------------------

if __name__ == '__main__':
    lat, lon = 43.3026, 5.3691
    start = datetime(2023, 1, 23, 5)
    end   = datetime(2023, 10, 23, 5)

    df = fetch_data(lat, lon, start, end,
                    email='saifiitp16@gmail.com',
                    identifier='cams_radiation')
    df = compute_pv_power(df)
    df_feat = prepare_features(df)
    train_forecast(df_feat)
