"""
ML Feature Logger - Machine Learning Dataset Builder
Logs ML-ready feature snapshots for every scan and symbol.
Designed to create a stable dataset for supervised or reinforcement learning.
"""

import os
import csv
from datetime import datetime, timezone


class MLFeatureLogger:
    """
    Logs ML-ready feature snapshots for every scan and symbol.
    Designed to create a stable dataset for supervised or reinforcement learning.
    """

    def __init__(self, base_path="ML"):
        self.base_path = base_path
        self.file_path = os.path.join(base_path, "features.csv")
        self.columns = [
            "timestamp",
            "symbol",
            "ema_fast",
            "ema_slow",
            "rsi",
            "atr",
            "adx",
            "macd_hist",
            "trend_bias",
            "volatility_regime",
            "spread_ratio",
            "session_state",
            "raw_signal",
            "signal_score",
            "reason"
        ]

        self._ensure_folder()
        self._ensure_header()

    def _ensure_folder(self):
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def _ensure_header(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def log_features(self, symbol: str, feature_dict: dict):
        """
        Writes a single ML feature row for a symbol.

        Args:
            symbol: symbol name
            feature_dict: extracted feature dictionary from engine
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        row = [
            timestamp,
            symbol,
            feature_dict.get("ema_fast"),
            feature_dict.get("ema_slow"),
            feature_dict.get("rsi"),
            feature_dict.get("atr"),
            feature_dict.get("adx"),
            feature_dict.get("macd_hist"),
            feature_dict.get("trend_bias"),
            feature_dict.get("volatility_regime"),
            feature_dict.get("spread_ratio"),
            feature_dict.get("session_state"),
            feature_dict.get("raw_signal"),
            feature_dict.get("signal_score"),
            feature_dict.get("reason"),
        ]

        try:
            with open(self.file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"[ML LOGGER] Failed to write row for {symbol}: {e}")

