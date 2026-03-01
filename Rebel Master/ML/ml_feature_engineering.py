"""
ml_feature_engineering.py

Feature engineering utilities for Rebel Master / ML models.

- Takes a DataFrame like trade_features.csv
- Adds engineered ML features (trend, volatility, momentum, regime, RR, etc.)
- Returns an enriched DataFrame ready for merging with labels and training.

Expected base columns (from trade_features.csv):
    ['timestamp', 'ticket', 'symbol', 'direction', 'entry_price', 'sl', 'tp',
     'volume', 'score', 'ema_fast', 'ema_slow', 'rsi', 'atr', 'adx',
     'macd_hist', 'trend_bias', 'volatility_regime', 'spread_ratio',
     'session_state', 'raw_signal', 'signal_score', 'reason']
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Convert timestamp column to pandas datetime, ignore errors."""
    return pd.to_datetime(series, errors="coerce")


def add_ml_features(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    """
    Add ML-ready engineered features to a features DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Original features dataframe (e.g. trade_features.csv loaded into pandas).
    symbol_col : str
        Column name for symbol/instrument (default: 'symbol').

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ML feature columns.
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # 1. Ensure timestamp is datetime for time-based features
    # -------------------------------------------------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = _safe_to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.NaT

    # Grouped helper for time series features
    if symbol_col in df.columns:
        grp = df.groupby(symbol_col, group_keys=False)
    else:
        grp = df

    # -------------------------------------------------------------------------
    # 2. Trend features (EMA-based)
    # -------------------------------------------------------------------------
    # Distances between EMAs and price
    if {"ema_fast", "ema_slow"}.issubset(df.columns):
        df["distance_fast_slow"] = df["ema_fast"] - df["ema_slow"]
    else:
        df["distance_fast_slow"] = np.nan

    if {"entry_price", "ema_fast"}.issubset(df.columns):
        df["distance_price_fast"] = df["entry_price"] - df["ema_fast"]
    else:
        df["distance_price_fast"] = np.nan

    if {"entry_price", "ema_slow"}.issubset(df.columns):
        df["distance_price_slow"] = df["entry_price"] - df["ema_slow"]
    else:
        df["distance_price_slow"] = np.nan

    # EMA "angles" = first difference per symbol
    if "ema_fast" in df.columns:
        df["ema_fast_angle"] = grp["ema_fast"].diff()
    else:
        df["ema_fast_angle"] = np.nan

    if "ema_slow" in df.columns:
        df["ema_slow_angle"] = grp["ema_slow"].diff()
    else:
        df["ema_slow_angle"] = np.nan

    # Trend strength = magnitude of EMA distance
    df["trend_strength"] = df["distance_fast_slow"].abs()

    # Trend alignment: price & EMA both on same side
    df["trend_alignment"] = np.where(
        (df["ema_fast"] > df["ema_slow"]) & (df["entry_price"] > df["ema_fast"]),
        1,
        np.where(
            (df["ema_fast"] < df["ema_slow"]) & (df["entry_price"] < df["ema_fast"]),
            -1,
            0,
        ),
    )

    # -------------------------------------------------------------------------
    # 3. Volatility features (ATR and change in ATR)
    # -------------------------------------------------------------------------
    if {"atr", "entry_price"}.issubset(df.columns):
        df["atr_ratio"] = df["atr"] / df["entry_price"].abs().replace(0, np.nan)
    else:
        df["atr_ratio"] = np.nan

    if "atr" in df.columns:
        df["volatility_delta"] = grp["atr"].diff()
        df["volatility_acceleration"] = grp["volatility_delta"].diff()
        # Volatility z-score per symbol
        df["volatility_zscore"] = grp["atr"].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        )
    else:
        df["volatility_delta"] = np.nan
        df["volatility_acceleration"] = np.nan
        df["volatility_zscore"] = np.nan

    # -------------------------------------------------------------------------
    # 4. Momentum features (MACD, RSI, ADX)
    # -------------------------------------------------------------------------
    if "macd_hist" in df.columns:
        df["macd_abs"] = df["macd_hist"].abs()
        df["macd_state"] = (df["macd_hist"] > 0).astype(int)
    else:
        df["macd_abs"] = np.nan
        df["macd_state"] = 0

    if "rsi" in df.columns:
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_midzone"] = ((df["rsi"] >= 45) & (df["rsi"] <= 55)).astype(int)
    else:
        df["rsi_overbought"] = 0
        df["rsi_oversold"] = 0
        df["rsi_midzone"] = 0

    if "adx" in df.columns:
        df["adx_trending"] = (df["adx"] > 25).astype(int)
    else:
        df["adx_trending"] = 0

    # -------------------------------------------------------------------------
    # 5. Market regime / time-of-day features
    # -------------------------------------------------------------------------
    # Session numeric (if already encoded as int, just copy)
    if "session_state" in df.columns:
        # If it's already numeric, keep it. If it's string-like, map a simple code.
        if np.issubdtype(df["session_state"].dtype, np.number):
            df["session_numeric"] = df["session_state"].astype(int)
        else:
            session_map = {
                "asia": 0,
                "london": 1,
                "ny": 2,
                "post": 3,
            }
            df["session_numeric"] = (
                df["session_state"].str.lower().map(session_map).fillna(-1).astype(int)
            )
    else:
        df["session_numeric"] = -1

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour.fillna(-1).astype(int)
    df["minute"] = df["timestamp"].dt.minute.fillna(-1).astype(int)
    df["day_of_week"] = df["timestamp"].dt.weekday.fillna(-1).astype(int)

    # -------------------------------------------------------------------------
    # 6. Spread / execution features
    # -------------------------------------------------------------------------
    # spread_ratio already exists and was very important in RF
    # Use it to create a volatility-adjusted spread feature.
    if {"spread_ratio", "atr"}.issubset(df.columns):
        df["spread_volatility_ratio"] = df["spread_ratio"] * df["atr"]
    else:
        df["spread_volatility_ratio"] = np.nan

    # Spread delta (change in spread_ratio per symbol)
    if "spread_ratio" in df.columns:
        df["spread_delta"] = grp["spread_ratio"].diff()
    else:
        df["spread_delta"] = np.nan

    # -------------------------------------------------------------------------
    # 7. Trade setup features: SL/TP structure, RR, volume normalization
    # -------------------------------------------------------------------------
    if {"entry_price", "sl"}.issubset(df.columns):
        df["sl_distance"] = (df["entry_price"] - df["sl"]).abs()
    else:
        df["sl_distance"] = np.nan

    if {"entry_price", "tp"}.issubset(df.columns):
        df["tp_distance"] = (df["tp"] - df["entry_price"]).abs()
    else:
        df["tp_distance"] = np.nan

    df["reward_risk_ratio"] = df["tp_distance"] / (df["sl_distance"] + 1e-9)

    # Normalize volume per symbol
    if "volume" in df.columns and symbol_col in df.columns:
        symbol_vol_med = df.groupby(symbol_col)["volume"].transform(
            lambda s: s.median() + 1e-9
        )
        df["volume_norm"] = df["volume"] / symbol_vol_med
    else:
        df["volume_norm"] = np.nan

    # -------------------------------------------------------------------------
    # 8. Direction encoding (optional helper)
    # -------------------------------------------------------------------------
    if "direction" in df.columns:
        # Check if already numeric (engine may pre-convert 1=long, -1=short)
        if np.issubdtype(df["direction"].dtype, np.number):
            df["direction_encoded"] = df["direction"].astype(int)
        else:
            # Encode long/short as 1 / -1 (anything else = 0)
            df["direction_encoded"] = np.where(
                df["direction"].str.lower() == "long",
                1,
                np.where(df["direction"].str.lower() == "short", -1, 0),
            )
    else:
        df["direction_encoded"] = 0

    return df


if __name__ == "__main__":
    # Example usage (for local testing):
    # df = pd.read_csv("trade_features.csv")
    # df_ml = add_ml_features(df)
    # df_ml.to_csv("trade_features_ml.csv", index=False)
    pass

