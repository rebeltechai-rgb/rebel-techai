"""
ML Label Generator - Step 7 Upgrade
Creates enhanced ML labels for every closed trade.
"""

import os
import csv
from datetime import datetime, timezone


class MLLabelGenerator:
    """
    Creates ML labels including:
    - outcome_class (win/loss)
    - reward_ratio
    - max_favorable_excursion (MFE)
    - max_adverse_excursion (MAE)
    - time_in_trade (minutes)
    - volatility_normalized_reward
    """

    def __init__(self, base_path="ML"):
        self.base_path = base_path
        self.file_path = os.path.join(base_path, "labels.csv")

        self.columns = [
            "timestamp",
            "ticket",
            "symbol",
            "direction",
            "entry_price",
            "exit_price",
            "sl_distance",
            "atr",
            "pnl",
            "rr",
            "norm_pnl",
            "label",
            "outcome_class",
            "reward_ratio",
            "mfe",
            "mae",
            "time_in_trade",
            "volatility_normalized_reward"
        ]

        self._ensure_folder()
        self._ensure_header()

    def _ensure_folder(self):
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def _ensure_header(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def log_label(
        self,
        ticket,
        symbol,
        direction,
        entry_price,
        exit_price,
        sl_distance,
        atr,
        mfe,
        mae,
        time_in_trade
    ):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # --- P&L calculation ---
        pnl = exit_price - entry_price if direction == "long" else entry_price - exit_price
        
        # --- Binary label (1=win, 0=loss) - better for ML ---
        label = 1 if pnl > 0 else 0
        
        # --- Reward-to-Risk (RR) ---
        rr = (pnl / sl_distance) if sl_distance > 0 else 0.0
        
        # --- Normalized P&L (pnl / atr) - volatility adjusted ---
        norm_pnl = (pnl / atr) if atr > 0 else 0.0

        # 1. Outcome class (string version for readability)
        outcome_class = "win" if pnl > 0 else "loss"

        # 2. Reward ratio (R/R-based outcome) - same as rr but kept for compatibility
        reward_ratio = rr

        # 3. Volatility-normalized reward (legacy field)
        if atr > 0:
            vol_norm_reward = reward_ratio / atr
        else:
            vol_norm_reward = 0

        # Write label row
        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                ticket,
                symbol,
                direction,
                round(entry_price, 6),
                round(exit_price, 6),
                round(sl_distance, 6),
                round(atr, 6),
                round(pnl, 6),
                round(rr, 4),
                round(norm_pnl, 4),
                label,
                outcome_class,
                round(reward_ratio, 4),
                round(mfe, 6),
                round(mae, 6),
                round(time_in_trade, 2),
                round(vol_norm_reward, 6)
            ])

        print(f"[ML_LABEL] Logged label for ticket {ticket} ({outcome_class}, RR={rr:.2f})")
