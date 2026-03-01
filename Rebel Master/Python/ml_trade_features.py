"""
ML Trade Features Logger - Logs features at TRADE ENTRY TIME with ticket ID.
This creates a proper link between features and trade outcomes (labels).

Features are captured at the moment a trade is opened, ensuring:
- Each feature row corresponds to exactly one trade
- Can be merged with labels.csv on 'ticket' column
"""

import os
import csv
from datetime import datetime, timezone


class MLTradeFeatureLogger:
    """
    Logs ML features at trade entry time with ticket ID.
    Creates trade_features.csv that can be joined with labels.csv on ticket.
    """

    def __init__(self, base_path=r"C:\Rebel Technologies\Rebel Master\ML"):
        self.base_path = base_path
        self.file_path = os.path.join(base_path, "trade_features.csv")
        self.columns = [
            "timestamp",
            "ticket",
            "symbol",
            "direction",
            "entry_price",
            "sl",
            "tp",
            "volume",
            "score",
            # ML Features
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
            print(f"[ML_TRADE_FEATURES] Created folder: {self.base_path}")

    def _ensure_header(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)
            print(f"[ML_TRADE_FEATURES] Created file: {self.file_path}")

    def log_trade_features(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        volume: float,
        score: int,
        features: dict
    ):
        """
        Log ML features at the moment a trade is opened.

        Args:
            ticket: Trade ticket number (unique ID)
            symbol: Trading symbol
            direction: 'long' or 'short'
            entry_price: Entry price
            sl: Stop loss price
            tp: Take profit price
            volume: Lot size
            score: Signal score
            features: Feature dictionary from scanner
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        row = [
            timestamp,
            ticket,
            symbol,
            direction,
            round(entry_price, 6),
            round(sl, 6),
            round(tp, 6),
            round(volume, 4),
            score,
            # ML Features
            features.get("ema_fast"),
            features.get("ema_slow"),
            features.get("rsi"),
            features.get("atr"),
            features.get("adx"),
            features.get("macd_hist", 0),
            features.get("trend_bias", "unknown"),
            features.get("volatility_regime", "unknown"),
            features.get("spread_ratio", 0),
            features.get("session_state", "unknown"),
            features.get("raw_signal", "none"),
            features.get("signal_score", 0),
            features.get("reason", "")
        ]

        try:
            with open(self.file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"[ML_TRADE_FEATURES] Logged features for ticket {ticket} ({symbol} {direction})")
        except Exception as e:
            print(f"[ML_TRADE_FEATURES] Failed to log for ticket {ticket}: {e}")

