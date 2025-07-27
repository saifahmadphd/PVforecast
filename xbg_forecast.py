"""
PV power forecasting with XGBoost (H-hour ahead) + lag scan + SHAP
------------------------------------------------------------------
* Forecast horizon set by H (default 36 h).
* Uses CAMS GHI + ERA5 air temperature via Open-Meteo archive.
* Adds utilities to:
  - scan cross-correlations vs. lag to suggest PV lags for the chosen H,
  - compute global SHAP importances and an optional beeswarm plot.

Run:
    python pv_forecast_xgb_clean_with_lag_scan_and_shap.py

Notes:
- Install SHAP first: `pip install shap` (or `conda install -c conda-forge shap`).
- If SHAP is missing, the script will skip SHAP plots and print a notice.
- Daylight-only filtering (ghi > 10 W/m^2) is used in the lag scan to avoid
  night-time correlations.
"""
from __future__ import annotations

# === Imports =================================================================
from datetime import datetime
import warnings
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error

from pvlib.iotools import get_cams
from pvlib.temperature import sapm_cell

# Optional SHAP
try:
    import shap  # type: ignore
    _HAVE_SHAP = True
except Exception:  # pragma: no cover
    _HAVE_SHAP = False

plt.rc("text", usetex=False)

# === Config ==================================================================
HORIZON_H = 36                    # forecast horizon (hours)
DEFAULT_LAGS = (12, 36, 60)       # fallback PV lags if scan is skipped/empty
ROLL_WIN = 2                      # rolling mean window (hours)
TRAIN_FRACTION = 0.90             # train/test split fraction
SEED = 42

# Lag-scan parameters
MAX_LAG_SCAN = 168                # scan up to this many hours back (1 week)
DAYLIGHT_GHI_MIN = 10.0           # W/m^2; used to mask nights in scan
N_TOP_PV_LAGS = 5                 # how many PV lags to keep from scan
MIN_ABS_CORR = 0.05               # minimal absolute correlation to accept
MIN_LAG_SPACING = 6               # minimal spacing (hours) between chosen lags

# SHAP parameters
SHAP_SAMPLE_ROWS = 4000           # subsample rows for SHAP to speed up
MAX_DISPLAY_BEEESWARM = 20

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
    js = requests.get(url, timeout=60).json()['hourly']
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

# === 3) Lag correlation scan ==================================================

def lag_correlation_scan(df: pd.DataFrame, H: int,
                         max_lag: int = MAX_LAG_SCAN,
                         cols: tuple[str, ...] = ('pv_power_kW', 'ghi', 'temp_air', 'cell_temp'),
                         method: str = 'pearson',
                         daylight_min: float | None = DAYLIGHT_GHI_MIN) -> pd.DataFrame:
    """Compute correlation between target (PV at t+H) and each variable lagged 1..max_lag.

    Returns a tidy DataFrame with columns: variable, lag_h, corr, abs_corr.
    """
    if 'pv_power_kW' not in df:
        raise ValueError("df must contain 'pv_power_kW'. Call compute_pv_power first.")

    target = df['pv_power_kW'].shift(-H)
    mask = target.notna()
    if daylight_min is not None and 'ghi' in df:
        mask &= df['ghi'] > float(daylight_min)

    rows: list[tuple[str, int, float | None]] = []
    for col in cols:
        s = df[col]
        for lag in range(1, max_lag + 1):
            corr = s.shift(lag)[mask].corr(target[mask], method=method)
            rows.append((col, lag, corr))
    out = pd.DataFrame(rows, columns=['variable', 'lag_h', 'corr']).dropna()
    out['abs_corr'] = out['corr'].abs()
    return out.sort_values('abs_corr', ascending=False)


def propose_pv_lags(scan: pd.DataFrame, n_top: int = N_TOP_PV_LAGS,
                    min_abs_corr: float = MIN_ABS_CORR,
                    min_spacing: int = MIN_LAG_SPACING) -> list[int]:
    """Pick PV lags from a scan, enforcing minimal spacing to avoid near-duplicates."""
    s = (scan.query("variable == 'pv_power_kW'")
              .sort_values('abs_corr', ascending=False))

    chosen: list[int] = []
    for _, row in s.iterrows():
        lag = int(row['lag_h'])
        if row['abs_corr'] < min_abs_corr:
            continue
        if all(abs(lag - c) >= min_spacing for c in chosen):
            chosen.append(lag)
        if len(chosen) >= n_top:
            break
    return sorted(chosen)

# === 4) Feature engineering ===================================================

def build_df_feat(df: pd.DataFrame,
                  H: int = HORIZON_H,
                  lags: tuple[int, ...] | list[int] = DEFAULT_LAGS,
                  roll_win: int = ROLL_WIN) -> tuple[pd.DataFrame, list[str], str]:
    """Create features/target for XGB. Returns (df_feat, feature_cols, target_col)."""
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

    # Lags of PV (selected)
    for lag in lags:
        df_feat[f'lag_{lag}h'] = df_feat['pv_power'].shift(lag)

    # Target
    target_col = f'target_{H}h'
    df_feat[target_col] = df_feat['pv_power'].shift(-H)

    # Drop rows with NaNs in any used column
    feature_cols = [c for c in df_feat.columns if c != target_col]
    df_feat = df_feat.dropna(subset=feature_cols + [target_col])

    return df_feat, feature_cols, target_col

# === 5) Train & evaluate XGB ==================================================

def train_xgb(df_feat: pd.DataFrame, feature_cols: list[str], target_col: str,
              train_frac: float = TRAIN_FRACTION,
              seed: int = SEED):
    X = df_feat[feature_cols]
    y = df_feat[target_col]

    split = int(len(df_feat) * train_frac)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_cols)

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
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        'y_true': y_test,
        'y_pred': y_pred,
        'rmse': rmse,
        'index': df_feat.index[split:],
        'bst': bst,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'split': split,
    }

# === 6) SHAP utilities ========================================================

def compute_shap(bst: xgb.Booster, X_test: pd.DataFrame,
                 feature_cols: list[str], sample_rows: int = SHAP_SAMPLE_ROWS):
    if not _HAVE_SHAP:
        warnings.warn("SHAP not available; skipping explanations.")
        return None, None, None

    # Optionally subsample for speed/memory
    if sample_rows is not None and len(X_test) > sample_rows:
        Xs = X_test.sample(sample_rows, random_state=SEED)
    else:
        Xs = X_test

    dm = xgb.DMatrix(Xs, feature_names=feature_cols)
    explainer = shap.TreeExplainer(bst)
    explanation = explainer(dm)
    shap_values = explanation.values  # (n_samples, n_features)
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = (pd.DataFrame({'feature': feature_cols, 'mean_abs_shap': mean_abs})
                .sort_values('mean_abs_shap', ascending=False)
                .reset_index(drop=True))
    return imp_df, explanation, Xs

# === 7) Plotting ==============================================================

def plot_results(idx, y_true, y_pred, title: str):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(idx, y_true, label='Observed PV', color='C0')
    ax1.plot(idx, y_pred, label='XGB Forecast', color='C1')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('PV Power (kW)')
    ax1.legend(loc='upper left', fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_corr_curves(scan: pd.DataFrame):
    for v in scan['variable'].unique():
        sub = scan[scan['variable'] == v].sort_values('lag_h')
        plt.figure(figsize=(7, 3))
        plt.plot(sub['lag_h'], sub['corr'], label=v)
        plt.axhline(0, linewidth=1)
        plt.xlabel('Lag (h)'); plt.ylabel('Correlation')
        plt.title(f'Corr(target t+H, {v}(t−lag))')
        plt.legend(); plt.tight_layout(); plt.show()

# === 8) Script entry-point ====================================================

if __name__ == '__main__':
    lat, lon = 43.3026, 5.3691
    start = datetime(2023, 1, 23, 5)
    end   = datetime(2023, 12, 23, 5)

    df = fetch_data(lat, lon, start, end,
                    email='saifiitp16@gmail.com',
                    identifier='cams_radiation')
    df = compute_pv_power(df)

    # --- Lag scan to suggest PV lags for this H ------------------------------
    print(f"Scanning lag correlations up to {MAX_LAG_SCAN} h for H={HORIZON_H} h ...")
    scan = lag_correlation_scan(df, H=HORIZON_H, max_lag=MAX_LAG_SCAN)
    pv_lags = propose_pv_lags(scan, n_top=N_TOP_PV_LAGS,
                              min_abs_corr=MIN_ABS_CORR,
                              min_spacing=MIN_LAG_SPACING)
    if pv_lags:
        print("Suggested PV lags (h) from scan:", pv_lags)
        use_lags = pv_lags
    else:
        print("Lag scan produced no candidates above threshold; using defaults:", DEFAULT_LAGS)
        use_lags = list(DEFAULT_LAGS)

    # Uncomment to visualize correlation vs. lag curves
    # plot_corr_curves(scan)

    # --- Build features with selected lags -----------------------------------
    df_feat, feature_cols, target_col = build_df_feat(df, H=HORIZON_H, lags=use_lags)

    # --- Train/evaluate -------------------------------------------------------
    res = train_xgb(df_feat, feature_cols, target_col)
    print(f"{HORIZON_H}-h-ahead RMSE (non-neg): {res['rmse']:.3f} kW")

    plot_results(res['index'], res['y_true'], res['y_pred'],
                 title=f"PV Forecast vs Temperature ({HORIZON_H} h ahead)")

    # --- SHAP explanations ----------------------------------------------------
    if _HAVE_SHAP:
        print("Computing SHAP importances (TreeExplainer)...")
        imp_df, explanation, Xs = compute_shap(res['bst'], res['X_test'], res['feature_cols'])
        if imp_df is not None:
            print("Top features by mean(|SHAP|):")
            print(imp_df.head(20).to_string(index=False))
            try:
                shap.plots.beeswarm(explanation, max_display=MAX_DISPLAY_BEEESWARM)
            except Exception as e:  # pragma: no cover
                warnings.warn(f"Beeswarm plot failed: {e}")
    else:
        print("SHAP not installed. Install with `pip install shap` to get explanations.")
