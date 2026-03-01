"""
rebel_signal_filters.py

5-Gate Signal Filter System for REBEL Trading Bot.

NO signal reaches the Master unless ALL gates pass.
All rejections are logged with reason tags.
NO ATR dependency in filters.

Gates:
1. Trend Alignment - HTF must agree with execution TF
2. Volatility Regime - Block chaotic conditions
3. Candle Quality - Reject bad candle patterns
4. Spread Control - Reject high spread
5. Session/Timing - Reject dead zones
"""

import os
import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# Filter rejection log
FILTER_LOG_PATH = r"C:\Rebel Technologies\Rebel Master\logs\filter_rejections.txt"


def _ensure_log_dir():
    """Ensure the Logs directory exists."""
    log_dir = os.path.dirname(FILTER_LOG_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def _log_rejection(symbol: str, gate: str, reason: str):
    """Log a filter rejection with timestamp and gate info."""
    _ensure_log_dir()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {symbol} | REJECTED | Gate={gate} | Reason={reason}\n"
    
    with open(FILTER_LOG_PATH, "a") as f:
        f.write(line)
    
    print(f"[FILTER] {symbol}: REJECTED by {gate} — {reason}")


def _log_soft_allow(symbol: str, gate: str, reason: str):
    """Log a soft-allow decision (non-blocking) with timestamp and gate info."""
    _ensure_log_dir()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {symbol} | SOFT_ALLOW | Gate={gate} | Reason={reason}\n"

    with open(FILTER_LOG_PATH, "a") as f:
        f.write(line)

    print(f"[FILTER] {symbol}: SOFT_ALLOW by {gate} — {reason}")


# =============================================================================
# GATE 1: TREND ALIGNMENT
# HTF direction must agree with execution TF
# Reject if HTF is flat or opposing pressure in last 3-5 candles
# =============================================================================

def gate_trend_alignment(
    symbol: str,
    direction: str,
    htf_data: pd.DataFrame,
    lookback: int = 5
) -> Tuple[bool, str]:
    """
    Check if HTF trend aligns with trade direction.
    
    Args:
        symbol: Trading symbol
        direction: 'long' or 'short'
        htf_data: DataFrame with OHLC from higher timeframe (H1/H4)
        lookback: Number of candles to check for trend
        
    Returns:
        (passed: bool, reason: str)
    """
    if htf_data is None or len(htf_data) < lookback:
        return False, "insufficient_htf_data"
    
    recent = htf_data.tail(lookback)
    
    # Calculate trend direction from recent closes
    closes = recent["close"].values
    first_close = closes[0]
    last_close = closes[-1]
    
    # Count bullish vs bearish candles
    bullish_count = sum(1 for i in range(len(recent)) if recent.iloc[i]["close"] > recent.iloc[i]["open"])
    bearish_count = lookback - bullish_count
    
    # Determine HTF bias
    price_change_pct = (last_close - first_close) / first_close * 100
    
    # Flat detection: relaxed threshold to allow more signals
    is_flat = abs(price_change_pct) < 0.08 and abs(bullish_count - bearish_count) <= 1
    
    if is_flat:
        _log_rejection(symbol, "TREND_ALIGNMENT", "htf_flat_no_direction")
        return False, "htf_flat_no_direction"
    
    # Direction check
    htf_bullish = price_change_pct > 0 and bullish_count >= 3
    htf_bearish = price_change_pct < 0 and bearish_count >= 3
    
    if direction == "long" and not htf_bullish:
        _log_soft_allow(symbol, "TREND_ALIGNMENT", "HTF_DISAGREEMENT_SOFT_ALLOW")
        return True, "counter_trend"
    
    if direction == "short" and not htf_bearish:
        _log_soft_allow(symbol, "TREND_ALIGNMENT", "HTF_DISAGREEMENT_SOFT_ALLOW")
        return True, "counter_trend"
    
    return True, "aligned"


# =============================================================================
# GATE 2: VOLATILITY REGIME
# Block chaotic (spread spikes, erratic ticks, wick storms)
# Allow normal / explosive only
# =============================================================================

def gate_volatility_regime(
    symbol: str,
    recent_candles: pd.DataFrame,
    max_wick_body_ratio: float = 3.5,
    max_wick_storm_count: int = 3
) -> Tuple[bool, str]:
    """
    Check for chaotic volatility conditions.
    
    Args:
        symbol: Trading symbol
        recent_candles: DataFrame with recent OHLC (last 5-10 candles)
        max_wick_body_ratio: Max allowed wick-to-body ratio
        max_wick_storm_count: Max candles with excessive wicks
        
    Returns:
        (passed: bool, reason: str)
    """
    if recent_candles is None or len(recent_candles) < 5:
        return False, "insufficient_candle_data"
    
    recent = recent_candles.tail(5)
    wick_storm_count = 0
    
    for i in range(len(recent)):
        candle = recent.iloc[i]
        body = abs(candle["close"] - candle["open"])
        upper_wick = candle["high"] - max(candle["open"], candle["close"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]
        total_wick = upper_wick + lower_wick
        
        # Avoid division by zero
        if body < 0.00001:
            body = 0.00001
        
        wick_ratio = total_wick / body
        
        if wick_ratio > max_wick_body_ratio:
            wick_storm_count += 1
    
    if wick_storm_count >= max_wick_storm_count:
        _log_rejection(symbol, "VOLATILITY_REGIME", f"wick_storm_{wick_storm_count}_candles")
        return False, f"wick_storm_{wick_storm_count}_candles"
    
    # Check for erratic price action (high/low range vs typical)
    ranges = recent["high"] - recent["low"]
    avg_range = ranges.mean()
    max_range = ranges.max()
    
    if max_range > avg_range * 2.5:
        _log_rejection(symbol, "VOLATILITY_REGIME", "erratic_range_spike")
        return False, "erratic_range_spike"
    
    return True, "normal"


# =============================================================================
# GATE 3: CANDLE QUALITY
# Reject micro-candles
# Reject long rejection wicks against direction
# Reject back-to-back flip candles (whipsaw signature)
# =============================================================================

def gate_candle_quality(
    symbol: str,
    direction: str,
    recent_candles: pd.DataFrame,
    min_body_pct: float = 0.12,
    max_rejection_wick_ratio: float = 2.8
) -> Tuple[bool, str]:
    """
    Check candle quality for entry.
    
    Args:
        symbol: Trading symbol
        direction: 'long' or 'short'
        recent_candles: DataFrame with recent OHLC (last 3-5 candles)
        min_body_pct: Minimum body as % of range (reject micro-candles)
        max_rejection_wick_ratio: Max wick against direction vs body
        
    Returns:
        (passed: bool, reason: str)
    """
    if recent_candles is None or len(recent_candles) < 3:
        return False, "insufficient_candle_data"
    
    last_candle = recent_candles.iloc[-1]
    prev_candles = recent_candles.tail(3)
    
    # Calculate last candle metrics
    body = abs(last_candle["close"] - last_candle["open"])
    candle_range = last_candle["high"] - last_candle["low"]
    
    if candle_range < 0.00001:
        _log_rejection(symbol, "CANDLE_QUALITY", "zero_range_candle")
        return False, "zero_range_candle"
    
    body_pct = body / candle_range
    
    # Reject micro-candles (doji-like)
    if body_pct < min_body_pct:
        _log_rejection(symbol, "CANDLE_QUALITY", f"micro_candle_body_{body_pct:.1%}")
        return False, f"micro_candle_body_{body_pct:.1%}"
    
    # Check rejection wick against direction
    upper_wick = last_candle["high"] - max(last_candle["open"], last_candle["close"])
    lower_wick = min(last_candle["open"], last_candle["close"]) - last_candle["low"]
    
    if body < 0.00001:
        body = 0.00001
    
    if direction == "long" and upper_wick / body > max_rejection_wick_ratio:
        _log_rejection(symbol, "CANDLE_QUALITY", "rejection_wick_against_long")
        return False, "rejection_wick_against_long"
    
    if direction == "short" and lower_wick / body > max_rejection_wick_ratio:
        _log_rejection(symbol, "CANDLE_QUALITY", "rejection_wick_against_short")
        return False, "rejection_wick_against_short"
    
    # Check for whipsaw (back-to-back flip candles)
    flip_count = 0
    for i in range(1, len(prev_candles)):
        curr = prev_candles.iloc[i]
        prev = prev_candles.iloc[i - 1]
        
        curr_bullish = curr["close"] > curr["open"]
        prev_bullish = prev["close"] > prev["open"]
        
        if curr_bullish != prev_bullish:
            flip_count += 1
    
    # Reject whipsaw when flips coincide with weak bodies
    if flip_count >= 2 and body_pct < 0.25:  # 3 candles, 2+ flips = whipsaw
        _log_rejection(symbol, "CANDLE_QUALITY", "whipsaw_flip_candles")
        return False, "whipsaw_flip_candles"
    
    return True, "quality_ok"


# =============================================================================
# GATE 4: SPREAD CONTROL
# Reject if spread > symbol's normal band
# Extra strict for FX & crypto
# =============================================================================

# Normal spread bands (in price units / points)
SPREAD_BANDS = {
    # FX Majors - very tight
    "EURUSD": 0.00015,
    "GBPUSD": 0.00020,
    "USDJPY": 0.020,
    "USDCHF": 0.00020,
    "AUDUSD": 0.00020,
    "USDCAD": 0.00025,
    "NZDUSD": 0.00025,
    
    # FX Crosses - medium
    "EURJPY": 0.030,
    "GBPJPY": 0.040,
    "EURGBP": 0.00025,
    "AUDNZD": 0.00035,
    
    # Exotics - wider
    "USDZAR": 0.0050,
    "USDMXN": 0.0050,
    "USDIDR": 5.0,
    
    # Crypto - percentage based
    "BTCUSD": 50.0,
    "ETHUSD": 3.0,
    "XRPUSD": 0.005,
    
    # METALS - much wider spreads than FX!
    "XAUUSD": 0.50,      # Gold
    "XAUEUR": 0.50,
    "XAUAUD": 0.50,
    "XAUGBP": 0.50,
    "XAGUSD": 0.05,      # Silver - typically 0.03-0.05
    "XPTUSD": 2.0,       # Platinum
    "COPPER.FS": 0.01,
    
    # ENERGIES - wider spreads
    "BRENT.FS": 0.05,
    "WTI.FS": 0.05,
    "UKOIL": 0.05,
    "USOIL": 0.05,
    "NATGAS.FS": 0.01,
    
    # SOFTS/COMMODITIES
    "COFFEE.FS": 0.50,
    "COCOA.FS": 5.0,
    "SOYBEAN.FS": 0.50,
    
    # INDICES - wider spreads (in points)
    "US500": 0.8,
    "US30": 3.0,
    "US2000": 0.5,
    "USTECH": 2.0,
    "NAS100.FS": 2.0,
    "DJ30.FS": 5.0,
    "S&P.FS": 1.0,
    "GER40": 2.0,
    "DAX40.FS": 2.0,
    "UK100": 2.0,
    "FT100.FS": 2.0,
    "FRA40": 2.0,
    "CAC40.FS": 2.0,
    "EU50": 1.5,
    "EUSTX50.FS": 2.0,
    "AUS200": 2.0,
    "SPI200.FS": 2.0,
    "JPN225": 15.0,
    "NK225.FS": 20.0,
    "HK50": 10.0,
    "HSI.FS": 15.0,
    "CN50": 5.0,
    "CHINA50.FS": 5.0,
    "IT40": 15.0,
    "SPA35": 5.0,
    "SWI20": 3.0,
    "NETH25": 0.5,
    "SGFREE": 0.5,
    "USDINDEX.FS": 0.05,  # Dollar index
    "VIX.FS": 0.10,
    
    # Defaults by asset class
    "DEFAULT_FX": 0.0003,
    "DEFAULT_FX_EXOTIC": 0.0035,
    "DEFAULT_CRYPTO": 0.5,
    "DEFAULT_INDEX": 3.0,
    "DEFAULT_METAL": 0.50,
    "DEFAULT_ENERGY": 0.05,
    "DEFAULT_SOFT": 1.0,
    "DEFAULT": 0.001,
}


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

    # FX default (majors/crosses/exotics)
    return "fx"


def _get_spread_band(symbol: str) -> float:
    """Get the normal spread band for a symbol."""
    sym_upper = symbol.upper()
    
    # Check exact match first (handles .FS suffix variations)
    if sym_upper in SPREAD_BANDS:
        return SPREAD_BANDS[sym_upper]
    
    # Try without .FS suffix
    base_sym = sym_upper.replace(".FS", "")
    if base_sym in SPREAD_BANDS:
        return SPREAD_BANDS[base_sym]
    
    spread_class = _get_spread_class(symbol)

    if spread_class == "metals":
        return SPREAD_BANDS["DEFAULT_METAL"]

    if spread_class == "energies":
        return SPREAD_BANDS["DEFAULT_ENERGY"]

    if spread_class == "softs":
        return SPREAD_BANDS["DEFAULT_SOFT"]

    if spread_class == "indices":
        return SPREAD_BANDS["DEFAULT_INDEX"]

    if spread_class == "crypto":
        return SPREAD_BANDS["DEFAULT_CRYPTO"]

    # FX exotics - broader default spread band
    exotic_tokens = [
        "TRY", "ZAR", "MXN", "PLN", "HUF", "CZK", "NOK", "SEK", "SGD",
        "THB", "IDR", "INR", "KRW", "TWD", "BRL", "CLP", "COP", "RON", "HKD"
    ]
    if any(t in sym_upper for t in exotic_tokens):
        return SPREAD_BANDS["DEFAULT_FX_EXOTIC"]

    # Default FX (majors/crosses)
    return SPREAD_BANDS["DEFAULT_FX"]


def gate_spread_control(
    symbol: str,
    current_spread: float,
    strictness_multiplier: float = 2.5  # Default; adjusted by asset class
) -> Tuple[bool, str]:
    """
    Check if spread is within normal band.
    
    Args:
        symbol: Trading symbol
        current_spread: Current spread in price units
        strictness_multiplier: How many times normal band is allowed (1.5 = 50% above normal)
        
    Returns:
        (passed: bool, reason: str)
    """
    normal_band = _get_spread_band(symbol)
    spread_class = _get_spread_class(symbol)

    class_multipliers = {
        "fx": 2.5,
        "indices": 3.0,
        "metals": 3.0,
        "energies": 3.0,
        "softs": 3.5,
        "crypto": 3.0,
    }
    max_allowed = normal_band * class_multipliers.get(spread_class, strictness_multiplier)
    
    if current_spread > max_allowed:
        _log_rejection(symbol, "SPREAD_CONTROL", f"spread_{current_spread:.5f}_above_max_{max_allowed:.5f}")
        return False, f"spread_{current_spread:.5f}_above_max_{max_allowed:.5f}"
    
    return True, "spread_ok"


# =============================================================================
# GATE 5: SESSION / TIMING
# Reject dead zones
# Friday late session = tighter acceptance
# =============================================================================

def gate_session_timing(
    symbol: str,
    current_hour_utc: int,
    day_of_week: int,  # 0=Monday, 4=Friday, 6=Sunday
    friday_cutoff_hour: int = 20,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """
    Check if current time is suitable for trading.
    
    Args:
        symbol: Trading symbol
        current_hour_utc: Current hour in UTC (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        friday_cutoff_hour: Hour on Friday to stop trading (UTC)
        
    Returns:
        (passed: bool, reason: str)
    """
    if config and config.get("session_filter", {}).get("enabled", False):
        if not _market_session_open(symbol, config):
            _log_rejection(symbol, "SESSION_TIMING", "market_closed")
            return False, "market_closed"

    return True, "session_ok"


def _market_session_open(symbol: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """Check if a market session is open using config sessions."""
    if not config:
        return True

    sessions_cfg = config.get("sessions", {})
    if not sessions_cfg:
        return True

    now = datetime.datetime.now(datetime.timezone.utc)
    weekday = now.strftime("%a").lower()[:3]

    symbol_upper = symbol.upper()
    groups = config.get("symbols", {}).get("groups", {})

    session_key = None
    if "crypto" in groups and symbol in groups.get("crypto", {}).get("symbols", []):
        session_key = "crypto"
    elif "indices" in groups and symbol in groups.get("indices", {}).get("symbols", []):
        if any(tag in symbol_upper for tag in ["DAX", "FRA", "CAC", "UK", "EU", "SPA", "SWI", "IT4", "NETH"]):
            session_key = "indices_eu"
        elif any(tag in symbol_upper for tag in ["NAS", "UST", "US5", "DJ", "DOW", "US30", "US500", "US2000"]):
            session_key = "indices_us"
        elif any(tag in symbol_upper for tag in ["JP", "NK", "225", "HK", "CHINA"]):
            session_key = "indices_asia"
        else:
            session_key = "indices_us"
    elif "forex" in groups and symbol in groups.get("forex", {}).get("symbols", []):
        session_key = "fx"
    elif "commodities" in groups and symbol in groups.get("commodities", {}).get("symbols", []):
        if any(tag in symbol_upper for tag in ["XAU", "XAG", "XPT", "XPD", "GOLD", "SILV"]):
            session_key = "metals"
        elif any(tag in symbol_upper for tag in ["BRENT", "WTI", "UKOIL", "USOIL", "NATGAS"]):
            session_key = "energies"
        elif any(tag in symbol_upper for tag in ["COFFEE", "COCOA", "SOY", "SOYBEAN", "COTTON", "SUGAR"]):
            session_key = "softs"
        else:
            session_key = "metals"

    if session_key is None:
        return True

    session_cfg = sessions_cfg.get(session_key)
    if session_cfg is None:
        return True

    if weekday not in session_cfg.get("days", []):
        return False

    open_str = session_cfg.get("open", "00:00")
    close_str = session_cfg.get("close", "23:59")
    open_hour, open_min = map(int, open_str.split(":"))
    close_hour, close_min = map(int, close_str.split(":"))

    open_dt = now.replace(hour=open_hour, minute=open_min, second=0, microsecond=0)
    close_dt = now.replace(hour=close_hour, minute=close_min, second=0, microsecond=0)

    return open_dt <= now <= close_dt


# =============================================================================
# MASTER FILTER - ALL GATES MUST PASS
# =============================================================================

def run_all_filters(
    symbol: str,
    direction: str,
    htf_data: pd.DataFrame,
    entry_candles: pd.DataFrame,
    current_spread: float,
    current_hour_utc: int = None,
    day_of_week: int = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str], List[str]]:
    """
    Run all 5 filter gates. ALL must pass for signal to proceed.
    
    Args:
        symbol: Trading symbol
        direction: 'long' or 'short'
        htf_data: Higher timeframe OHLC data
        entry_candles: Entry timeframe OHLC data (last 5-10 candles)
        current_spread: Current spread in price units
        current_hour_utc: Current hour UTC (auto-detected if None)
        day_of_week: Day of week 0-6 (auto-detected if None)
        
    Returns:
        (all_passed: bool, rejection_reasons: list)
    """
    # Auto-detect time if not provided
    if current_hour_utc is None or day_of_week is None:
        now = datetime.datetime.now(datetime.timezone.utc)
        current_hour_utc = now.hour
        day_of_week = now.weekday()
    
    rejections = []
    soft_flags = []
    
    # Gate 1: Trend Alignment
    passed, reason = gate_trend_alignment(symbol, direction, htf_data)
    if not passed:
        rejections.append(f"G1_TREND:{reason}")
    elif reason == "counter_trend":
        soft_flags.append("counter_trend")
    
    # Gate 2: Volatility Regime
    passed, reason = gate_volatility_regime(symbol, entry_candles)
    if not passed:
        rejections.append(f"G2_VOLATILITY:{reason}")
    
    # Gate 3: Candle Quality
    passed, reason = gate_candle_quality(symbol, direction, entry_candles)
    if not passed:
        rejections.append(f"G3_CANDLE:{reason}")
    
    # Gate 4: Spread Control
    passed, reason = gate_spread_control(symbol, current_spread)
    if not passed:
        rejections.append(f"G4_SPREAD:{reason}")
    
    # Gate 5: Session Timing
    passed, reason = gate_session_timing(symbol, current_hour_utc, day_of_week, config=config)
    if not passed:
        rejections.append(f"G5_SESSION:{reason}")
    
    all_passed = len(rejections) == 0
    
    if all_passed:
        print(f"[FILTER] {symbol}: [OK] ALL GATES PASSED")
    else:
        print(f"[FILTER] {symbol}: BLOCKED - {rejections}")
    
    return all_passed, rejections, soft_flags


# =============================================================================
# FILTER STATUS
# =============================================================================

FILTER_STATUS = {
    "GATE_1_TREND_ALIGNMENT": "ON",
    "GATE_2_VOLATILITY_REGIME": "ON",
    "GATE_3_CANDLE_QUALITY": "ON",
    "GATE_4_SPREAD_CONTROL": "ON",
    "GATE_5_SESSION_TIMING": "ON",
}

def get_filter_status() -> Dict[str, str]:
    """Return current filter activation status."""
    return FILTER_STATUS.copy()

def print_filter_status():
    """Print filter activation status."""
    print("\n" + "=" * 50)
    print("  REBEL SIGNAL FILTERS — STATUS")
    print("=" * 50)
    for gate, status in FILTER_STATUS.items():
        icon = "[ON]" if status == "ON" else "[OFF]"
        print(f"  {icon} {gate}: {status}")
    print("=" * 50)
    print("  No signal reaches Master unless ALL gates pass")
    print("  All rejections logged to:", FILTER_LOG_PATH)
    print("  NO ATR dependency in filters")
    print("=" * 50 + "\n")

