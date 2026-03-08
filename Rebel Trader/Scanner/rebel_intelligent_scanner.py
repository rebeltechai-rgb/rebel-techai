"""
REBEL Intelligent Scanner - Lite (TA-Only)
=========================================
Standalone scanner with:
  - Pluggable broker adapter (MT5 by default)
  - Multi-timeframe technical analysis
  - Adaptive confidence thresholds
  - Continuous scanning loop with configurable interval
  - Entry, SL, TP calculation
  - Console logging

Note: This lite version mirrors the full scanner behavior but has no OpenAI option.
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import yaml
import numpy as np
import pandas as pd

from broker_adapters import get_adapter

# ============================================================================
#   CONFIGURATION PATHS (LOCAL TO SCANNER PACKAGE)
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CONFIG_PATH = os.path.join(BASE_DIR, "Config", "master_config.yaml")
SYMBOL_LIST_PATH = os.path.join(BASE_DIR, "Config", "symbol_lists.yaml")
LOG_PATH = os.path.join(BASE_DIR, "logs", "intelligent_scanner.log")


# ============================================================================
#   TECHNICAL ANALYSIS UTILITIES
# ============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI calculation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_val = tr.rolling(window=period, min_periods=period).mean()
    return atr_val


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """MACD indicator."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def compute_bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return {"upper": upper, "middle": sma, "lower": lower}


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_smooth = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smooth)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
    adx_val = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_val


def compute_slope(series: pd.Series, length: int = 10) -> float:
    """Linear regression slope over last N points."""
    if len(series) < length:
        return 0.0
    y = series.iloc[-length:].values
    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    m = result[0][0]
    return float(m)


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """Stochastic Oscillator."""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(window=d_period).mean()
    return {"k": k, "d": d}


# ============================================================================
#   REBEL INTELLIGENT SCANNER CLASS
# ============================================================================

class RebelIntelligentScanner:
    """
    Standalone TA-driven market scanner with:
      - Multi-timeframe technical analysis
      - Adaptive confidence thresholds
      - Entry/SL/TP calculation
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.connected = False
        self.adapter = get_adapter(self.config)

        # Scanner settings from config
        is_cfg = self.config.get("intelligent_scanner", {})

        self.enabled: bool = is_cfg.get("enabled", True)
        self.timeframes: List[str] = is_cfg.get("timeframes", ["M5", "M15", "H1"])
        self.bars: int = is_cfg.get("bars", 400)
        self.scan_interval: int = is_cfg.get("scan_interval", 300)  # 5 minutes default
        self.market_open_max_tick_age_sec: int = is_cfg.get("market_open_max_tick_age_sec", 300)
        self.show_closed_summary: bool = is_cfg.get("show_closed_summary", False)
        self.show_ta_reasoning: bool = is_cfg.get("show_ta_reasoning", True)

        # Confidence & mode (TA-only)
        self.base_min_confidence: int = is_cfg.get("min_confidence", 55)
        requested_mode = is_cfg.get("mode", "ta_only")
        self.mode: str = "ta_only" if requested_mode != "ta_only" else requested_mode
        self.use_ai: bool = False

        # Risk settings
        self.risk_percent: float = is_cfg.get("risk_percent", 2.0)
        self.sl_atr_multiplier: float = is_cfg.get("sl_atr_multiplier", 2.0)
        self.tp_atr_multiplier: float = is_cfg.get("tp_atr_multiplier", 3.0)

        # Symbols
        self.symbols: List[str] = []

        # Stats tracking
        self.scan_count = 0
        self.signal_count = 0

    # ========================================================================
    #   BROKER CONNECTION
    # ========================================================================

    def connect(self) -> bool:
        """Initialize broker connection."""
        if self.connected:
            return True
        if not self.adapter.connect(self.config):
            print(f"[BROKER] Connection failed: {self.adapter.last_error()}")
            return False

        account = self.adapter.account_info()
        if account:
            print(f"[BROKER] Connected: {account.login} @ {account.server}")
            print(f"[BROKER] Balance: {account.balance:.2f} {account.currency}")

        self.connected = True
        return True

    def disconnect(self):
        """Shutdown broker connection."""
        if self.connected:
            self.adapter.shutdown()
            self.connected = False
            print("[BROKER] Disconnected")

    # ========================================================================
    #   SYMBOL MANAGEMENT
    # ========================================================================

    def _normalize_symbol(self, name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    def _resolve_symbol(self, config_symbol: str, all_symbols: list) -> Optional[str]:
        """Resolve config symbol to broker symbol (handles suffixes like .a, .m, .fs)."""
        sym_map = {self._normalize_symbol(s.name): s.name for s in all_symbols}
        base = self._normalize_symbol(config_symbol)

        # Exact match
        if base in sym_map:
            return sym_map[base]

        # Find matches with suffixes, prefer non-.sa (standard over Select)
        matches = [s.name for s in all_symbols if self._normalize_symbol(s.name).startswith(base)]
        if not matches:
            return None

        # Prefer standard account symbols (not .sa)
        standard = [m for m in matches if not m.lower().endswith('.sa')]
        return min(standard, key=len) if standard else matches[0]

    def load_symbols(self) -> List[str]:
        """Load symbols from configuration and resolve to broker format."""
        config_symbols = []

        # Try master config first
        cfg_symbols = self.config.get("symbols", {})
        groups = cfg_symbols.get("groups", {})

        for _, group_cfg in groups.items():
            if isinstance(group_cfg, dict):
                if not group_cfg.get("enabled", True):
                    continue
                group_symbols = group_cfg.get("symbols", [])
            elif isinstance(group_cfg, list):
                group_symbols = group_cfg
            else:
                continue

            if isinstance(group_symbols, list):
                config_symbols.extend(group_symbols)

        # Fallback to symbol_lists.yaml
        if not config_symbols and os.path.exists(SYMBOL_LIST_PATH):
            with open(SYMBOL_LIST_PATH, "r") as f:
                data = yaml.safe_load(f) or {}
            groups = data.get("groups", {})
            for _, lst in groups.items():
                if isinstance(lst, list):
                    config_symbols.extend(lst)

        # Remove duplicates
        config_symbols = list(dict.fromkeys(config_symbols))

        # Resolve to broker symbols
        all_symbols = self.adapter.symbols_get()
        if all_symbols:
            self.symbols = []
            for sym in config_symbols:
                resolved = self._resolve_symbol(sym, all_symbols)
                if resolved:
                    self.symbols.append(resolved)
            print(f"[SCANNER] Resolved {len(self.symbols)} symbols to broker format")
            if not self.symbols and config_symbols:
                self.symbols = config_symbols
                print("[SCANNER] Warning: resolution returned 0 symbols; using config symbols as-is")
        else:
            self.symbols = config_symbols

        return self.symbols

    def ensure_symbol(self, symbol: str) -> bool:
        """Ensure symbol is available in broker terminal."""
        if not self.adapter.symbol_select(symbol, True):
            return False
        info = self.adapter.symbol_info(symbol)
        return info is not None and info.visible

    def is_market_open(self, symbol: str) -> bool:
        """Skip symbols without recent ticks (likely closed)."""
        if self.market_open_max_tick_age_sec <= 0:
            return True
        tick = self.adapter.symbol_info_tick(symbol)
        if tick is None or not getattr(tick, "time", 0):
            return False
        return (time.time() - tick.time) <= self.market_open_max_tick_age_sec

    # ========================================================================
    #   DATA FETCHING
    # ========================================================================

    def get_historical_data(self, symbol: str, timeframe: str, bars: int = 400) -> Optional[pd.DataFrame]:
        """Fetch historical OHLC data from broker."""
        tf = self.adapter.get_timeframe(timeframe)
        rates = self.adapter.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def _fetch_mtf_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured timeframes."""
        data = {}
        for tf in self.timeframes:
            df = self.get_historical_data(symbol, tf, self.bars)
            if df is not None and not df.empty:
                data[tf] = df
        return data

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current bid/ask price."""
        tick = self.adapter.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {"bid": tick.bid, "ask": tick.ask, "last": tick.last}

    # ========================================================================
    #   TECHNICAL ANALYSIS
    # ========================================================================

    def _analyze_trend_mtf(self, mtf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Multi-timeframe trend analysis."""
        per_tf = {}
        votes = 0

        for tf, df in mtf_data.items():
            close = df["close"]
            ema20 = compute_ema(close, 20)
            ema50 = compute_ema(close, 50)

            if len(ema20) < 60 or len(ema50) < 60:
                continue

            ema20_last = float(ema20.iloc[-1])
            ema50_last = float(ema50.iloc[-1])
            ema50_slope = compute_slope(ema50, length=40)

            if ema20_last > ema50_last and ema50_slope > 0:
                per_tf[tf] = "uptrend"
                votes += 1
            elif ema20_last < ema50_last and ema50_slope < 0:
                per_tf[tf] = "downtrend"
                votes -= 1
            else:
                per_tf[tf] = "sideways"

        if votes >= 2:
            overall = "strong_up"
            trend_score = 25
        elif votes == 1:
            overall = "weak_up"
            trend_score = 15
        elif votes == 0:
            overall = "sideways"
            trend_score = 5
        elif votes == -1:
            overall = "weak_down"
            trend_score = 15
        else:
            overall = "strong_down"
            trend_score = 25

        adx_val = 0
        for tf in ("H1", "M15", "M5"):
            if tf in mtf_data:
                adx_series = compute_adx(mtf_data[tf])
                if not adx_series.empty:
                    adx_val = float(adx_series.iloc[-1])
                break

        return {"per_tf": per_tf, "overall": overall, "trend_score": trend_score, "adx": adx_val}

    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """RSI and MACD momentum analysis."""
        rsi_series = compute_rsi(df["close"], 14)
        rsi_last = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50

        macd_data = compute_macd(df["close"])
        macd_hist = float(macd_data["histogram"].iloc[-1]) if not macd_data["histogram"].empty else 0

        # Stochastic
        stoch = compute_stochastic(df)
        stoch_k = float(stoch["k"].iloc[-1]) if not stoch["k"].empty else 50

        if rsi_last >= 65:
            mood = "overbought_bullish"
            score = 15
        elif rsi_last >= 55:
            mood = "bullish"
            score = 12
        elif rsi_last <= 35:
            mood = "oversold_bearish"
            score = 15
        elif rsi_last <= 45:
            mood = "bearish"
            score = 12
        else:
            mood = "neutral"
            score = 5

        # Adjust score based on MACD confirmation
        if (mood == "bullish" or mood == "overbought_bullish") and macd_hist > 0:
            score += 3
        elif (mood == "bearish" or mood == "oversold_bearish") and macd_hist < 0:
            score += 3

        return {
            "rsi": round(rsi_last, 2),
            "macd_histogram": round(macd_hist, 6),
            "stochastic_k": round(stoch_k, 2),
            "mood": mood,
            "momentum_score": min(score, 20)
        }

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ATR-based volatility regime analysis."""
        atr_series = compute_atr(df, 14)
        last = float(atr_series.iloc[-1]) if not atr_series.empty else 0
        median = float(atr_series.median()) if not atr_series.empty else 0

        if median == 0 or np.isnan(median):
            regime = "normal"
            score = 5
        else:
            ratio = last / median
            if ratio > 1.6:
                regime = "expanding"
                score = 12
            elif ratio < 0.7:
                regime = "contracting"
                score = 3
            else:
                regime = "normal"
                score = 7

        return {
            "atr": round(last, 6),
            "atr_median": round(median, 6),
            "regime": regime,
            "volatility_score": score
        }

    def _analyze_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Price structure analysis (breakout/range/pullback zones)."""
        close = df["close"]
        high_n = close.rolling(window=50, min_periods=20).max()
        low_n = close.rolling(window=50, min_periods=20).min()

        last_close = float(close.iloc[-1])
        last_high = float(high_n.iloc[-1])
        last_low = float(low_n.iloc[-1])

        if np.isnan(last_high) or np.isnan(last_low):
            return {"bias": "unknown", "structure_score": 5}

        span = last_high - last_low
        if span <= 0:
            return {"bias": "unknown", "structure_score": 5}

        pos = (last_close - last_low) / span

        # Bollinger position
        bb = compute_bollinger(close)
        bb_upper = float(bb["upper"].iloc[-1])
        bb_lower = float(bb["lower"].iloc[-1])
        bb_position = "middle"
        if last_close > bb_upper:
            bb_position = "above_upper"
        elif last_close < bb_lower:
            bb_position = "below_lower"

        if pos >= 0.8:
            bias = "near_high_breakout_zone"
            score = 12
        elif pos <= 0.2:
            bias = "near_low_breakdown_zone"
            score = 12
        elif 0.4 <= pos <= 0.6:
            bias = "mid_range_chop"
            score = 5
        else:
            bias = "pullback_zone"
            score = 8

        return {
            "bias": bias,
            "position_in_range": round(pos, 2),
            "bollinger_position": bb_position,
            "structure_score": score
        }

    # ========================================================================
    #   ADAPTIVE CONFIDENCE
    # ========================================================================

    def _adaptive_min_confidence(self, trend_overall: str, vol_regime: str) -> int:
        """Adjust min confidence based on market conditions."""
        base = self.base_min_confidence
        delta = 0

        if trend_overall in ("strong_up", "strong_down"):
            if vol_regime in ("normal", "expanding"):
                delta -= 5
        if trend_overall == "sideways" and vol_regime == "expanding":
            delta += 10
        if vol_regime == "contracting":
            delta -= 5

        return max(30, min(85, base + delta))

    def _build_direction_and_confidence(
        self,
        trend_overall: str,
        momentum_mood: str,
        structure_bias: str,
        scores: Dict[str, int]
    ) -> Dict[str, Any]:
        """Combine component scores into direction and confidence."""
        trend_score = scores.get("trend_score", 0)
        momentum_score = scores.get("momentum_score", 0)
        volatility_score = scores.get("volatility_score", 0)
        structure_score = scores.get("structure_score", 0)

        raw_total = trend_score + momentum_score + volatility_score + structure_score
        # Max theoretical around 25 + 20 + 12 + 12 = 69 -> scale to 0-100
        base_confidence = int(np.clip((raw_total / 69) * 100, 0, 100))

        direction = None

        if trend_overall in ("strong_up", "weak_up"):
            if "bearish" not in momentum_mood:
                direction = "long"
        elif trend_overall in ("strong_down", "weak_down"):
            if "bullish" not in momentum_mood:
                direction = "short"
        else:
            # Sideways: only trade extremes
            if structure_bias == "near_high_breakout_zone" and "bullish" in momentum_mood:
                direction = "long"
            elif structure_bias == "near_low_breakdown_zone" and "bearish" in momentum_mood:
                direction = "short"

        return {"direction": direction, "base_confidence": base_confidence}

    def _build_ta_reasoning(
        self,
        trend_info: Dict[str, Any],
        momentum_info: Dict[str, Any],
        vol_info: Dict[str, Any],
        struct_info: Dict[str, Any],
        direction: str,
        base_confidence: int,
        min_confidence: int
    ) -> List[str]:
        """Build human-readable TA reasons for a signal."""
        reasons: List[str] = []

        trend_overall = trend_info.get("overall", "unknown")
        momentum_mood = momentum_info.get("mood", "neutral")
        vol_regime = vol_info.get("regime", "normal")
        structure_bias = struct_info.get("bias", "unknown")

        if direction == "long":
            reasons.append(f"Trend alignment bullish ({trend_overall}).")
            reasons.append(f"Momentum supports long bias ({momentum_mood}).")
        elif direction == "short":
            reasons.append(f"Trend alignment bearish ({trend_overall}).")
            reasons.append(f"Momentum supports short bias ({momentum_mood}).")

        if structure_bias != "unknown":
            reasons.append(f"Structure context: {structure_bias.replace('_', ' ')}.")

        if vol_regime != "normal":
            reasons.append(f"Volatility regime: {vol_regime} (ATR).")

        reasons.append(f"Confidence: {base_confidence}% (min {min_confidence}%).")
        return reasons

    # ========================================================================
    #   LOT SIZE CALCULATION
    # ========================================================================

    def _calc_lot(self, symbol: str, entry_price: float, sl_price: float) -> float:
        """Calculate lot size based on risk percentage and stop loss distance."""
        account = self.adapter.account_info()
        if account is None:
            raise RuntimeError("Failed to get account info from broker")

        symbol_info = self.adapter.symbol_info(symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbol info not found for {symbol}")

        # Risk amount in money
        risk_amount = account.balance * (self.risk_percent / 100.0)

        # Stop loss distance in price
        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            raise ValueError("SL distance must be greater than zero")

        # Convert SL distance into number of ticks
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        if tick_size <= 0 or tick_value <= 0:
            raise RuntimeError(f"Invalid tick settings for {symbol}")

        ticks = sl_distance / tick_size
        loss_per_lot = ticks * tick_value

        if loss_per_lot <= 0:
            raise RuntimeError(f"Invalid loss_per_lot for {symbol}")

        lot = risk_amount / loss_per_lot

        # Normalize to broker constraints
        vol_min = symbol_info.volume_min
        vol_max = symbol_info.volume_max
        vol_step = symbol_info.volume_step
        if vol_step <= 0:
            vol_step = 0.01

        lot = round(lot / vol_step) * vol_step
        lot = max(vol_min, min(vol_max, lot))

        return lot

    # ========================================================================
    #   ENTRY/SL/TP CALCULATION
    # ========================================================================

    def _calculate_trade_levels(self, symbol: str, direction: str, atr: float) -> Dict[str, float]:
        """Calculate entry, SL, and TP levels."""
        price_data = self.get_current_price(symbol)
        if not price_data:
            return {}

        symbol_info = self.adapter.symbol_info(symbol)
        if not symbol_info:
            return {}

        digits = symbol_info.digits

        if direction == "long":
            entry = price_data["ask"]
            sl = entry - (atr * self.sl_atr_multiplier)
            tp = entry + (atr * self.tp_atr_multiplier)
        else:  # short
            entry = price_data["bid"]
            sl = entry + (atr * self.sl_atr_multiplier)
            tp = entry - (atr * self.tp_atr_multiplier)

        try:
            lot = self._calc_lot(symbol, entry, sl)
        except Exception:
            lot = symbol_info.volume_min

        return {
            "entry": round(entry, digits),
            "sl": round(sl, digits),
            "tp": round(tp, digits),
            "lot": lot,
            "risk_reward": round(self.tp_atr_multiplier / self.sl_atr_multiplier, 2)
        }

    # ========================================================================
    #   MAIN SCAN METHOD
    # ========================================================================

    def scan_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Full analysis for a single symbol."""
        if not self.enabled:
            return None

        if not self.ensure_symbol(symbol):
            return None
        if not self.is_market_open(symbol):
            return None

        mtf_data = self._fetch_mtf_data(symbol)
        if not mtf_data:
            return None

        # Main timeframe for detailed analysis
        tf_main = "M15" if "M15" in mtf_data else list(mtf_data.keys())[0]
        main_df = mtf_data[tf_main]

        # Technical analysis components
        trend_info = self._analyze_trend_mtf(mtf_data)
        momentum_info = self._analyze_momentum(main_df)
        vol_info = self._analyze_volatility(main_df)
        struct_info = self._analyze_structure(main_df)

        # Adaptive confidence threshold
        adaptive_min_conf = self._adaptive_min_confidence(
            trend_overall=trend_info["overall"],
            vol_regime=vol_info["regime"]
        )

        # Combine scores
        scores = {
            "trend_score": trend_info["trend_score"],
            "momentum_score": momentum_info["momentum_score"],
            "volatility_score": vol_info["volatility_score"],
            "structure_score": struct_info["structure_score"]
        }

        dir_conf = self._build_direction_and_confidence(
            trend_overall=trend_info["overall"],
            momentum_mood=momentum_info["mood"],
            structure_bias=struct_info["bias"],
            scores=scores
        )

        direction = dir_conf["direction"]
        base_conf = dir_conf["base_confidence"]
        final_conf = base_conf

        ai_block = {
            "enabled": False,
            "ai_confidence": None,
            "reasoning": [],
            "direction_override": None,
            "key_levels": {},
            "risk_assessment": "unknown"
        }

        # Final filter
        raw_direction = direction
        raw_confidence = final_conf
        if final_conf < adaptive_min_conf or direction is None:
            direction = None

        # Calculate trade levels if we have a signal
        trade_levels = {}
        ta_reasoning = []
        if direction:
            trade_levels = self._calculate_trade_levels(symbol, direction, vol_info["atr"])
            ta_reasoning = self._build_ta_reasoning(
                trend_info=trend_info,
                momentum_info=momentum_info,
                vol_info=vol_info,
                struct_info=struct_info,
                direction=direction,
                base_confidence=base_conf,
                min_confidence=adaptive_min_conf,
            )

        result = {
            "symbol": symbol,
            "direction": direction,
            "raw_direction": raw_direction,
            "base_confidence": base_conf,
            "final_confidence": final_conf,
            "raw_confidence": raw_confidence,
            "min_confidence": adaptive_min_conf,
            "trend": trend_info,
            "momentum": momentum_info,
            "volatility": vol_info,
            "structure": struct_info,
            "scores": scores,
            "ai": ai_block,
            "ta_reasoning": ta_reasoning,
            "trade_levels": trade_levels,
            "timestamp": datetime.now().isoformat()
        }

        return result

    def scan_all(self) -> List[Dict[str, Any]]:
        """Scan all loaded symbols."""
        results = []
        signals = []
        raw_signals = []
        blocked_by_conf = 0
        blocked_by_direction = 0
        skipped_closed = 0
        open_symbols = []

        print(f"\n{'='*70}")
        print(f"REBEL INTELLIGENT SCANNER - Scan #{self.scan_count + 1}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {self.mode.upper()} | AI: OFF")
        for symbol in self.symbols:
            if self.is_market_open(symbol):
                open_symbols.append(symbol)
            else:
                skipped_closed += 1

        print(
            f"Symbols: {len(open_symbols)} open / {len(self.symbols)} total | "
            f"Min Confidence: {self.base_min_confidence}%"
        )
        print(f"{'='*70}")

        for symbol in open_symbols:
            try:
                res = self.scan_symbol(symbol)
                if res:
                    results.append(res)
                    if res.get("raw_direction"):
                        raw_signals.append(res)
                    if res["direction"]:
                        signals.append(res)
                    elif res.get("raw_direction") and res.get("raw_confidence", 0) < res.get("min_confidence", 0):
                        blocked_by_conf += 1
                    else:
                        blocked_by_direction += 1
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")

        self.scan_count += 1
        self.signal_count += len(signals)

        # Sort by confidence
        signals = sorted(signals, key=lambda r: r.get("final_confidence", 0), reverse=True)

        # Print signals
        if signals:
            print(f"\n[SIGNALS] Found: {len(signals)}")
            print("-" * 70)
            for sig in signals:
                sym = sig["symbol"]
                direction = sig["direction"].upper()
                conf = sig["final_confidence"]
                trend = sig["trend"]["overall"]
                levels = sig.get("trade_levels", {})

                arrow = "[+]" if direction == "LONG" else "[-]"
                print(f"{arrow} {sym:10} | {direction:5} | {conf:3d}% | Trend: {trend}")

                if levels:
                    print(
                        f"    Entry: {levels.get('entry', 'N/A')} | SL: {levels.get('sl', 'N/A')} | "
                        f"TP: {levels.get('tp', 'N/A')} | Lot: {levels.get('lot', 'N/A')}"
                    )

                if self.show_ta_reasoning and sig.get("ta_reasoning"):
                    for idx, reason in enumerate(sig["ta_reasoning"]):
                        prefix = "    TA:" if idx == 0 else "       "
                        print(f"{prefix} {reason}")

                if sig["ai"]["reasoning"]:
                    print(f"    AI: {sig['ai']['reasoning'][0][:60]}...")
                print()
        else:
            print("\n[--] No signals this scan")
            if raw_signals:
                raw_sorted = sorted(raw_signals, key=lambda r: r.get("raw_confidence", 0), reverse=True)
                print(f"[INFO] Candidates with direction: {len(raw_signals)} | Blocked by confidence: {blocked_by_conf}")
                print("[INFO] Top 5 candidates (raw):")
                for sig in raw_sorted[:5]:
                    sym = sig["symbol"]
                    direction = sig["raw_direction"].upper()
                    conf = sig.get("raw_confidence", 0)
                    min_conf = sig.get("min_confidence", 0)
                    trend = sig["trend"]["overall"]
                    print(f"    {sym:10} | {direction:5} | {conf:3d}% (min {min_conf}%) | Trend: {trend}")
            else:
                print(f"[INFO] No directional candidates. Blocked by direction rules: {blocked_by_direction}")
        if skipped_closed and self.show_closed_summary:
            print(f"[INFO] Skipped (market closed or stale ticks): {skipped_closed}")

        print(f"\nTotal scans: {self.scan_count} | Total signals: {self.signal_count}")
        print(f"{'='*70}\n")

        return results

    # ========================================================================
    #   MAIN RUN LOOP
    # ========================================================================

    def run(self, continuous: bool = True):
        """Main scanning loop."""
        print("\n" + "=" * 70)
        print("  REBEL INTELLIGENT SCANNER LITE")
        print("  TA-Driven Market Analysis Engine")
        print("=" * 70)

        if not self.connect():
            print("[FATAL] Could not connect to broker")
            return

        self.load_symbols()
        if not self.symbols:
            print("[FATAL] No symbols loaded")
            self.disconnect()
            return

        print(f"\n[CONFIG] Mode: {self.mode}")
        print(f"[CONFIG] Timeframes: {self.timeframes}")
        print(f"[CONFIG] Symbols: {len(self.symbols)}")
        print(f"[CONFIG] Scan interval: {self.scan_interval}s")
        print(f"[CONFIG] Risk: {self.risk_percent}% | SL: {self.sl_atr_multiplier}x ATR | TP: {self.tp_atr_multiplier}x ATR")

        print("\n[STATUS] Scanner running... Press Ctrl+C to stop\n")

        try:
            while True:
                self.scan_all()

                if not continuous:
                    break

                print(f"[WAIT] Next scan in {self.scan_interval}s...")
                time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            print("\n[STOP] Shutdown requested by user")
        finally:
            self.disconnect()


# ============================================================================
#   STANDALONE ENTRY POINT
# ============================================================================

def load_config() -> Dict[str, Any]:
    """Load master configuration."""
    if not os.path.exists(MASTER_CONFIG_PATH):
        print(f"[WARN] Config not found: {MASTER_CONFIG_PATH}")
        return {}
    with open(MASTER_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="REBEL Intelligent Scanner Lite - TA-Driven Market Analysis")
    parser.add_argument("--once", action="store_true", help="Run single scan and exit")
    parser.add_argument("--interval", type=int, default=None, help="Scan interval in seconds")
    parser.add_argument("--min_confidence", type=int, default=None, help="Override min confidence (0-100)")
    parser.add_argument("--show_ta_reasons", action="store_true", help="Show TA reasoning per signal")
    args = parser.parse_args()

    # Load config
    config = load_config()

    # Override config with CLI args
    if "intelligent_scanner" not in config:
        config["intelligent_scanner"] = {}

    if args.interval:
        config["intelligent_scanner"]["scan_interval"] = args.interval
    if args.min_confidence is not None:
        config["intelligent_scanner"]["min_confidence"] = args.min_confidence
    if args.show_ta_reasons:
        config["intelligent_scanner"]["show_ta_reasoning"] = True

    # Force TA-only mode in lite
    config["intelligent_scanner"]["mode"] = "ta_only"
    config["intelligent_scanner"]["use_ai"] = False

    # Create and run scanner
    scanner = RebelIntelligentScanner(config)
    scanner.run(continuous=not args.once)
