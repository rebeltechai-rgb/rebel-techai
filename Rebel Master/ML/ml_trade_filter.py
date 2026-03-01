"""
ml_trade_filter.py

ML-based trade filter for Rebel Master & future bots.

Responsibilities:
- Load a trained sklearn model (RandomForest, GradientBoosting, etc.)
- Take a candidate trade's features (same structure as trade_features rows)
- Add engineered ML features (using ml_feature_engineering.add_ml_features)
- Run the ML model to get win/loss probabilities
- Apply hard filters (spread, volatility, RR, etc.)
- Return a decision: ACCEPT / REJECT + reasons
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from ml_feature_engineering import add_ml_features


# ---------------------------------------------------------------------
# 1. Config: thresholds & toggles for the ML filter
# ---------------------------------------------------------------------

@dataclass
class MLFilterConfig:
    # ML probability thresholds
    min_win_prob: float = 0.55          # minimum P(win) to allow trade
    max_loss_prob: float = 0.65         # optional: reject if P(loss) too high

    # Hard filters from feature importance
    max_spread_ratio: float = 0.0035    # reject if spread_ratio above this
    max_spread_ratio_by_class: Dict[str, float] = field(default_factory=dict)
    max_spread_vol_ratio: float = 0.25  # spread_volatility_ratio guard
    max_atr_ratio: float = 0.02         # volatility too wild if above this

    # Reward:Risk filters
    min_reward_risk: float = 1.2        # reject if RR < 1.2
    max_sl_distance_atr: float = 3.0    # reject if SL > 3x ATR (too wide)

    # Regime / session filters
    block_midzone_rsi: bool = False     # optionally avoid RSI 45–55 chop
    allowed_sessions: Optional[List[int]] = field(
        default_factory=lambda: [0, 1, 2]
    )  # 0=Asia,1=London,2=NY; -1/post filtered out

    # Debug / logging
    return_reasons: bool = True         # include reasons for decision


# ---------------------------------------------------------------------
# 2. ML Filter class
# ---------------------------------------------------------------------

class MLTradeFilter:
    def __init__(self, model_path: str, config: Optional[MLFilterConfig] = None):
        """
        model_path: path to a joblib-saved sklearn model (e.g. GradientBoosting, RF)
        """
        loaded = joblib_load(model_path)
        
        # Handle both formats: direct model OR dict with 'model' key
        if isinstance(loaded, dict) and 'model' in loaded:
            self.model = loaded['model']
            print(f"[ML] Loaded model from dict (version: {loaded.get('version', 'unknown')})")
        else:
            self.model = loaded
            
        self.config = config or MLFilterConfig()
        
        # Load feature names from corresponding features file
        # e.g. model_rf_v2.joblib -> model_rf_v2_features.txt
        features_file = model_path.replace(".joblib", "_features.txt")
        self.feature_names = []
        try:
            with open(features_file, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    # Handle "feature,importance" format (take just the feature name)
                    if ',' in line:
                        line = line.split(',')[0].strip()
                    self.feature_names.append(line)
            print(f"[ML] Loaded {len(self.feature_names)} feature names from {features_file}")
        except FileNotFoundError:
            print(f"[ML] Warning: Feature names file not found: {features_file}")
            self.feature_names = []

    # ---- public API ----

    def evaluate_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single candidate trade.

        candidate: dict with keys like:
            'timestamp', 'symbol', 'direction', 'entry_price', 'sl', 'tp',
            'volume', 'ema_fast', 'ema_slow', 'rsi', 'atr', 'adx',
            'macd_hist', 'trend_bias', 'volatility_regime',
            'spread_ratio', 'session_state', 'raw_signal', 'signal_score', 'reason'
        (basically same fields as a row in trade_features.csv)

        Returns dict:
            {
              'decision': 'ACCEPT' or 'REJECT',
              'win_prob': float,
              'loss_prob': float,
              'reasons': [list of strings]
            }
        """
        # Wrap in DataFrame so we can reuse feature engine
        base_df = pd.DataFrame([candidate])

        # Add ML features (same as training)
        feat_df = add_ml_features(base_df)

        # Drop non-model columns (timestamp/symbol/reason/ticket)
        drop_cols = ["timestamp", "symbol", "reason", "ticket"]
        for col in drop_cols:
            if col in feat_df.columns:
                feat_df = feat_df.drop(columns=[col])

        # Handle any categorical leftovers and NaNs
        X = self._prepare_features_for_model(feat_df)

        # Get probabilities
        proba = self.model.predict_proba(X)[0]
        loss_prob = float(proba[0])  # assuming class 0 = loss, 1 = win
        win_prob = float(proba[1])

        reasons: List[str] = []
        decision = "ACCEPT"

        # ML probability logic
        if win_prob < self.config.min_win_prob:
            decision = "REJECT"
            reasons.append(f"win_prob {win_prob:.2f} below min {self.config.min_win_prob:.2f}")

        if loss_prob > self.config.max_loss_prob:
            decision = "REJECT"
            reasons.append(f"loss_prob {loss_prob:.2f} above max {self.config.max_loss_prob:.2f}")

        # Hard filters using engineered features
        decision, reasons = self._apply_hard_filters(
            feat_df.iloc[0],
            decision,
            reasons,
            symbol=candidate.get("symbol")
        )

        result = {
            "decision": decision,
            "win_prob": win_prob,
            "loss_prob": loss_prob,
        }
        if self.config.return_reasons:
            result["reasons"] = reasons

        return result

    # ---- internals ----

    def _prepare_features_for_model(self, df: pd.DataFrame) -> np.ndarray:
        X = df.copy()

        # Replace inf, nan
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)

        # Align features exactly with training columns (same order, same names)
        # Use self.feature_names loaded from features file, or model's feature_names_in_ if available
        expected_cols = []
        if self.feature_names:
            expected_cols = self.feature_names
        elif hasattr(self.model, 'feature_names_in_'):
            expected_cols = list(self.model.feature_names_in_)
        
        if expected_cols:
            # Add missing columns with 0
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0.0
            # Remove extra columns not in training
            extra_cols = [c for c in X.columns if c not in expected_cols]
            if extra_cols:
                X = X.drop(columns=extra_cols)
            # Reorder to match training order exactly
            X = X[expected_cols]

        # Convert to numpy array to avoid sklearn warning about feature names
        return X.values

    def _apply_hard_filters(
        self,
        row: pd.Series,
        decision: str,
        reasons: List[str],
        symbol: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        return _apply_hard_filters_impl(
            row=row,
            decision=decision,
            reasons=reasons,
            symbol=symbol,
            config=self.config,
        )


def _get_spread_class(symbol: str) -> str:
    """Classify symbol for spread limits."""
    sym_upper = symbol.upper()

    # METALS detection
    if any(m in sym_upper for m in ["XAU", "XAG", "XPT", "GOLD", "SILVER", "COPPER"]):
        return "metals"

    # ENERGIES detection
    if any(e in sym_upper for e in ["BRENT", "WTI", "OIL", "CRUDE", "NATGAS", "GAS"]):
        return "energies"

    # SOFTS/COMMODITIES detection
    if any(s in sym_upper for s in ["COFFEE", "COCOA", "SUGAR", "COTTON", "WHEAT", "CORN", "SOY"]):
        return "softs"

    # INDICES detection
    idx_keywords = [
        "US500", "US30", "US2000", "USTECH", "NAS", "DJ30", "S&P", "SPX",
        "DAX", "GER40", "UK100", "FT100", "FTSE", "FRA40", "CAC",
        "EU50", "EUSTX", "STOXX", "AUS200", "SPI200", "ASX",
        "JPN225", "NK225", "NIKKEI", "HK50", "HSI", "HANG",
        "CN50", "CHINA", "IT40", "SPA35", "SWI20", "NETH25",
        "SGFREE", "USDINDEX", "VIX", "INDEX"
    ]
    if any(i in sym_upper for i in idx_keywords):
        return "indices"

    # CRYPTO detection
    crypto_tokens = [
        "BTC", "ETH", "XRP", "LTC", "ADA", "SOL", "DOG", "DOT", "XLM", "BCH",
        "AVAX", "AAVE", "UNI", "SUSHI", "COMP", "CRV", "LRC", "MANA", "SAND",
        "BAT", "BNB", "KSM", "XTZ", "LNK"
    ]
    if any(c in sym_upper for c in crypto_tokens):
        return "crypto"

    exotic_tokens = [
        "TRY", "ZAR", "MXN", "PLN", "HUF", "CZK", "NOK", "SEK", "SGD",
        "THB", "IDR", "INR", "KRW", "TWD", "BRL", "CLP", "COP", "RON", "HKD"
    ]
    if any(t in sym_upper for t in exotic_tokens):
        return "fx_exotic"

    return "fx"

def _apply_hard_filters_impl(
    row: pd.Series,
    decision: str,
    reasons: List[str],
    symbol: Optional[str] = None,
    config: Optional[MLFilterConfig] = None,
) -> Tuple[str, List[str]]:
    cfg = config or MLFilterConfig()

    spread_ratio = float(row.get("spread_ratio", 0.0) or 0.0)
    spread_volatility_ratio = float(row.get("spread_volatility_ratio", 0.0) or 0.0)
    atr_ratio = float(row.get("atr_ratio", 0.0) or 0.0)
    reward_risk_ratio = float(row.get("reward_risk_ratio", 0.0) or 0.0)
    sl_distance = float(row.get("sl_distance", 0.0) or 0.0)
    atr = float(row.get("atr", 0.0) or 0.0)
    rsi_midzone = int(row.get("rsi_midzone", 0))
    session_numeric = int(row.get("session_numeric", -1))

    # 1) Spread filters
    max_spread_ratio = cfg.max_spread_ratio
    if cfg.max_spread_ratio_by_class and symbol:
        spread_class = _get_spread_class(symbol)
        max_spread_ratio = cfg.max_spread_ratio_by_class.get(spread_class, max_spread_ratio)
    if spread_ratio > max_spread_ratio:
        decision = "REJECT"
        reasons.append(
            f"spread_ratio {spread_ratio:.5f} > max {max_spread_ratio:.5f}"
        )

    if spread_volatility_ratio > cfg.max_spread_vol_ratio:
        decision = "REJECT"
        reasons.append(
            f"spread_volatility_ratio {spread_volatility_ratio:.3f} > max {cfg.max_spread_vol_ratio:.3f}"
        )

    # 2) Volatility filters
    if atr_ratio > cfg.max_atr_ratio:
        decision = "REJECT"
        reasons.append(
            f"atr_ratio {atr_ratio:.4f} > max {cfg.max_atr_ratio:.4f}"
        )

    # 3) Reward:Risk filters
    if reward_risk_ratio < cfg.min_reward_risk:
        decision = "REJECT"
        reasons.append(
            f"reward_risk_ratio {reward_risk_ratio:.2f} < min {cfg.min_reward_risk:.2f}"
        )

    if atr > 0:
        sl_atr = sl_distance / atr
        if sl_atr > cfg.max_sl_distance_atr:
            decision = "REJECT"
            reasons.append(
                f"SL {sl_atr:.2f} ATR > max {cfg.max_sl_distance_atr:.2f} ATR"
            )

    # 4) RSI chop filter (optional)
    if cfg.block_midzone_rsi and rsi_midzone == 1:
        decision = "REJECT"
        reasons.append("RSI midzone (45–55), chop regime blocked")

    # 5) Session filter
    if cfg.allowed_sessions is not None and session_numeric not in cfg.allowed_sessions:
        decision = "REJECT"
        reasons.append(f"session {session_numeric} not in allowed {cfg.allowed_sessions}")

    return decision, reasons


# ---------------------------------------------------------------------
# 3. Shadow Mode Helper
# ---------------------------------------------------------------------

def shadow_mode_decision(filter_obj: MLTradeFilter, candidate: dict) -> dict:
    """
    Runs ML filter in SHADOW mode (never blocks trades).
    Returns decision + probabilities + reasons, but engine is unaffected.
    
    Use this during training data collection to see what the model
    WOULD have decided, without actually blocking any trades.
    
    Args:
        filter_obj: MLTradeFilter instance with loaded model
        candidate: Trade candidate dict (same as evaluate_candidate)
        
    Returns:
        dict with decision, probabilities, reasons, and shadow_mode flags
    """
    result = filter_obj.evaluate_candidate(candidate)
    result["shadow_mode"] = True
    result["decision_override"] = "NO_BLOCK"
    return result

