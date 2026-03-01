"""
REBEL AI Core - Central Intelligence Module

This module provides the main intelligence engine for the REBEL system.
It is designed so that:

- The MAIN SYSTEM (engine) uses FULL MODE ("engine") → deeper analysis.
- The SCANNER uses LITE MODE ("scanner") → similar intelligence but shallower,
  never exceeding the engine's reasoning depth.

The AI here is pure Python (TA-based) with an "intelligence_score" and
"adaptive_threshold" system. If you later want to add GPT calls, we plug them
in here under a flag, without changing engine or scanner code.
"""

from dataclasses import dataclass
from typing import Dict, Any, Literal
import numpy as np
import pandas as pd


ModeType = Literal["engine", "scanner"]


@dataclass
class RebelAIConfig:
    min_bars: int = 200
    engine_depth_weight: float = 1.0      # full power for engine
    scanner_depth_weight: float = 0.65    # scanner gets ~65% of full depth
    base_threshold: int = 60              # default minimum score (was 55 - stricter)
    max_threshold: int = 80               # cap for very choppy markets (was 75)
    min_threshold: int = 50               # floor for very clean markets (was 45)


class RebelAI:
    """
    Core intelligence engine.

    Usage:
        ai = RebelAI()
        result = ai.analyze(symbol, df_m5, df_m15, df_h1, mode="engine")

    Returns:
        {
            "symbol": "XAUUSD",
            "mode": "engine" or "scanner",
            "regime": "strong_up" | "weak_up" | "sideways" | "weak_down" | "strong_down",
            "volatility": "contracting" | "normal" | "expanding",
            "momentum": "bullish" | "neutral" | "bearish",
            "bias": "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell",
            "intelligence_score": 0-100,
            "adaptive_threshold": 0-100,
            "reasons": [list of short strings]
        }
    """

    def __init__(self, config: RebelAIConfig | None = None):
        self.config = config or RebelAIConfig()

    # ---------------------- helper methods ----------------------

    def _safe_tail(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if len(df) < n:
            return df.copy()
        return df.tail(n).copy()

    def _slope(self, series: pd.Series, lookback: int = 20) -> float:
        s = series.dropna()
        if len(s) < lookback:
            return 0.0
        tail = s.tail(lookback)
        x = np.arange(len(tail))
        y = tail.values
        # simple linear regression slope
        denom = (x - x.mean()) ** 2
        denom_sum = denom.sum()
        if denom_sum == 0:
            return 0.0
        slope = ((x - x.mean()) * (y - y.mean())).sum() / denom_sum
        return float(slope)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if df is None or len(df) < period + 1:
            return 0.0
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        trs = []
        for i in range(1, len(df)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            trs.append(tr)
        if len(trs) < period:
            return 0.0
        return float(pd.Series(trs).tail(period).mean())

    def _rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        if df is None or len(df) < period + 1:
            return 50.0
        close = df["close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.dropna().iloc[-1])

    def _ema(self, series: pd.Series, period: int) -> float:
        if series is None or len(series) < period:
            return float(series.iloc[-1]) if len(series) else 0.0
        return float(series.ewm(span=period, adjust=False).mean().iloc[-1])

    # ---------------------- core analysis ----------------------

    def analyze(
        self,
        symbol: str,
        df_m5: pd.DataFrame,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame,
        mode: ModeType = "engine",
    ) -> Dict[str, Any]:
        """
        Main intelligence entrypoint.

        mode = "engine" → full depth
        mode = "scanner" → reduced depth (never smarter than system)
        """
        depth_weight = (
            self.config.engine_depth_weight if mode == "engine" else self.config.scanner_depth_weight
        )

        reasons: list[str] = []
        insufficient_data = (
            df_m5 is None or len(df_m5) < self.config.min_bars or
            df_m15 is None or len(df_m15) < self.config.min_bars or
            df_h1 is None or len(df_h1) < self.config.min_bars
        )

        if insufficient_data:
            return {
                "symbol": symbol,
                "mode": mode,
                "regime": "unknown",
                "volatility": "unknown",
                "momentum": "neutral",
                "bias": "neutral",
                "intelligence_score": 0,
                "adaptive_threshold": self.config.base_threshold,
                "reasons": ["insufficient data"],
            }

        # Use only the last N bars for regime / vol / momentum
        m5 = self._safe_tail(df_m5, 200)
        m15 = self._safe_tail(df_m15, 200)
        h1 = self._safe_tail(df_h1, 200)

        # Trend regime from H1
        h1_close = h1["close"]
        ema_fast = self._ema(h1_close, 20)
        ema_slow = self._ema(h1_close, 50)
        ema_diff = ema_fast - ema_slow
        slope_h1 = self._slope(h1_close, lookback=30)

        if ema_diff > 0 and slope_h1 > 0:
            if abs(ema_diff) > abs(h1_close.iloc[-1]) * 0.0015:
                regime = "strong_up"
                reasons.append("H1 strong uptrend (EMA20>EMA50 + positive slope)")
            else:
                regime = "weak_up"
                reasons.append("H1 weak uptrend")
        elif ema_diff < 0 and slope_h1 < 0:
            if abs(ema_diff) > abs(h1_close.iloc[-1]) * 0.0015:
                regime = "strong_down"
                reasons.append("H1 strong downtrend (EMA20<EMA50 + negative slope)")
            else:
                regime = "weak_down"
                reasons.append("H1 weak downtrend")
        else:
            regime = "sideways"
            reasons.append("H1 sideways / mixed trend")

        # Volatility regime from ATR on M15
        atr_recent = self._atr(m15, period=14)
        # compute median ATR over last 100 bars
        atr_series = []
        if len(m15) > 30:
            hi = m15["high"].values
            lo = m15["low"].values
            cl = m15["close"].values
            for i in range(1, len(m15)):
                tr = max(
                    hi[i] - lo[i],
                    abs(hi[i] - cl[i - 1]),
                    abs(lo[i] - cl[i - 1]),
                )
                atr_series.append(tr)
        if len(atr_series) >= 30:
            atr_median = float(pd.Series(atr_series).tail(100).median())
        else:
            atr_median = atr_recent

        if atr_median == 0:
            volatility = "normal"
        else:
            ratio = atr_recent / atr_median
            if ratio < 0.8:
                volatility = "contracting"
                reasons.append("Volatility contracting (ATR below median)")
            elif ratio > 1.3:
                volatility = "expanding"
                reasons.append("Volatility expanding (ATR above median)")
            else:
                volatility = "normal"
                reasons.append("Volatility normal")

        # Momentum from RSI across TFs
        rsi_m5 = self._rsi(m5, period=14)
        rsi_m15 = self._rsi(m15, period=14)
        rsi_h1 = self._rsi(h1, period=14)
        rsi_avg = (rsi_m5 + rsi_m15 + rsi_h1) / 3.0

        if rsi_avg > 60:
            momentum = "bullish"
            reasons.append(f"Momentum bullish (avg RSI ~ {rsi_avg:.1f})")
        elif rsi_avg < 40:
            momentum = "bearish"
            reasons.append(f"Momentum bearish (avg RSI ~ {rsi_avg:.1f})")
        else:
            momentum = "neutral"
            reasons.append(f"Momentum neutral (avg RSI ~ {rsi_avg:.1f})")

        # Bias from regime + momentum combo
        if regime == "strong_up" and momentum == "bullish":
            bias = "strong_buy"
        elif regime in ("strong_up", "weak_up") and momentum != "bearish":
            bias = "buy"
        elif regime == "sideways":
            bias = "neutral"
        elif regime in ("strong_down", "weak_down") and momentum != "bullish":
            bias = "sell"
        elif regime == "strong_down" and momentum == "bearish":
            bias = "strong_sell"
        else:
            bias = "neutral"

        reasons.append(f"Bias computed as {bias} from regime={regime}, momentum={momentum}")

        # Intelligence score (0–100)
        score = 50.0

        # Regime contribution
        if regime == "strong_up" or regime == "strong_down":
            score += 15 * depth_weight
        elif regime in ("weak_up", "weak_down"):
            score += 8 * depth_weight
        else:  # sideways
            score -= 10 * depth_weight

        # Momentum contribution
        if momentum == "bullish" or momentum == "bearish":
            score += 10 * depth_weight
        elif momentum == "neutral":
            score -= 5 * depth_weight

        # Volatility contribution
        if volatility == "normal":
            score += 5 * depth_weight
        elif volatility == "contracting":
            score += 3 * depth_weight   # good for scalps
        elif volatility == "expanding":
            score -= 5 * depth_weight   # more danger, be stricter

        # Clamp score
        score = max(0.0, min(100.0, score))

        # Adaptive threshold derived from regime + vol
        threshold = float(self.config.base_threshold)

        if regime in ("strong_up", "strong_down"):
            threshold -= 5 * depth_weight
        elif regime == "sideways":
            threshold += 5 * depth_weight

        if volatility == "expanding":
            threshold += 5 * depth_weight
        elif volatility == "contracting":
            threshold -= 2 * depth_weight

        # Clamp threshold within global bounds
        threshold = max(self.config.min_threshold, min(self.config.max_threshold, threshold))

        return {
            "symbol": symbol,
            "mode": mode,
            "regime": regime,
            "volatility": volatility,
            "momentum": momentum,
            "bias": bias,
            "intelligence_score": int(round(score)),
            "adaptive_threshold": int(round(threshold)),
            "reasons": reasons,
        }

