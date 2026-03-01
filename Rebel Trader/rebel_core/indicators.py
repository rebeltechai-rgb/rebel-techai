"""
REBEL RULES-BASED AI TRADING SYSTEM
Module: Technical Indicators
Pure mathematical computations - NO ML/AI
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timezone


def compute_all_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute all technical indicators for trading decisions.
    
    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
    
    Returns:
        Dictionary with all computed indicators
    """
    if df.empty or len(df) < 50:
        return _empty_indicators()
    
    result = {}
    
    # === MOVING AVERAGES ===
    result['ema9'] = _ema(df['close'], 9)
    result['ema21'] = _ema(df['close'], 21)
    result['ema50'] = _ema(df['close'], 50)
    result['sma20'] = df['close'].rolling(20).mean().iloc[-1]
    result['sma50'] = df['close'].rolling(50).mean().iloc[-1]
    result['sma200'] = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else None
    
    # === MOMENTUM INDICATORS ===
    result['rsi14'] = _rsi(df['close'], 14)
    result['rsi7'] = _rsi(df['close'], 7)
    
    # === MACD ===
    macd_data = _macd(df['close'], 12, 26, 9)
    result['macd_line'] = macd_data['macd']
    result['macd_signal'] = macd_data['signal']
    result['macd_histogram'] = macd_data['histogram']
    
    # === TREND STRENGTH ===
    result['adx14'] = _adx(df, 14)
    
    # === VOLATILITY ===
    result['atr14'] = _atr(df, 14)
    result['atr_percent'] = (result['atr14'] / df['close'].iloc[-1]) * 100 if df['close'].iloc[-1] > 0 else 0
    
    # === BOLLINGER BANDS ===
    bb = _bollinger_bands(df['close'], 20, 2)
    result['bb_upper'] = bb['upper']
    result['bb_middle'] = bb['middle']
    result['bb_lower'] = bb['lower']
    result['bb_width'] = bb['width']
    result['bb_position'] = bb['position']  # Where price is within bands (0-1)
    
    # === CURRENT PRICE DATA ===
    result['close'] = float(df['close'].iloc[-1])
    result['open'] = float(df['open'].iloc[-1])
    result['high'] = float(df['high'].iloc[-1])
    result['low'] = float(df['low'].iloc[-1])
    result['volume'] = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0
    
    # === DERIVED CLASSIFICATIONS ===
    result['trend'] = _classify_trend(df, result)
    result['trend_strength'] = _classify_trend_strength(result['adx14'])
    result['volatility'] = _classify_volatility(result['atr_percent'])
    result['momentum'] = _classify_momentum(result['rsi14'], result['macd_histogram'])
    result['market_structure'] = _analyze_structure(df)
    result['patterns'] = _detect_patterns(df)
    result['risk_level'] = _calculate_risk_level(result)
    
    return result


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def _ema(series: pd.Series, period: int) -> float:
    """Calculate Exponential Moving Average."""
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def _rsi(series: pd.Series, period: int = 14) -> float:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    result = rsi.iloc[-1]
    return float(result) if not pd.isna(result) else 50.0


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """Calculate MACD indicator."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': float(macd_line.iloc[-1]),
        'signal': float(signal_line.iloc[-1]),
        'histogram': float(histogram.iloc[-1])
    }


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average Directional Index."""
    if len(df) < period + 1:
        return 25.0
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Smoothing
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(window=period).mean()
    
    result = adx.iloc[-1]
    return float(result) if not pd.isna(result) else 25.0


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(df) < period + 1:
        return 0.0
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    result = atr.iloc[-1]
    return float(result) if not pd.isna(result) else 0.0


def _bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
    """Calculate Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    current_price = series.iloc[-1]
    middle = sma.iloc[-1]
    upper_val = upper.iloc[-1]
    lower_val = lower.iloc[-1]
    
    # Position within bands (0 = at lower, 1 = at upper)
    band_range = upper_val - lower_val
    position = (current_price - lower_val) / band_range if band_range > 0 else 0.5
    
    return {
        'upper': float(upper_val),
        'middle': float(middle),
        'lower': float(lower_val),
        'width': float(band_range / middle * 100) if middle > 0 else 0,
        'position': float(max(0, min(1, position)))
    }


# =============================================================================
# CLASSIFICATIONS
# =============================================================================

def _classify_trend(df: pd.DataFrame, indicators: Dict[str, Any]) -> str:
    """
    Classify trend direction based on EMA relationships and price action.
    
    Returns: "STRONG_UP", "UP", "SIDEWAYS", "DOWN", "STRONG_DOWN"
    """
    ema9 = indicators['ema9']
    ema21 = indicators['ema21']
    ema50 = indicators['ema50']
    close = indicators['close']
    
    # Calculate slope (5-candle momentum)
    if len(df) >= 5:
        slope = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
    else:
        slope = 0
    
    # EMA alignment
    ema_bullish = ema9 > ema21 > ema50
    ema_bearish = ema9 < ema21 < ema50
    
    # Price above/below EMAs
    price_above_all = close > ema9 and close > ema21 and close > ema50
    price_below_all = close < ema9 and close < ema21 and close < ema50
    
    if ema_bullish and price_above_all and slope > 0.003:
        return "STRONG_UP"
    elif ema9 > ema21 and slope > 0.001:
        return "UP"
    elif ema_bearish and price_below_all and slope < -0.003:
        return "STRONG_DOWN"
    elif ema9 < ema21 and slope < -0.001:
        return "DOWN"
    else:
        return "SIDEWAYS"


def _classify_trend_strength(adx: float) -> str:
    """
    Classify trend strength based on ADX.
    
    Returns: "STRONG", "MODERATE", "WEAK", "NO_TREND"
    """
    if adx >= 40:
        return "STRONG"
    elif adx >= 25:
        return "MODERATE"
    elif adx >= 20:
        return "WEAK"
    else:
        return "NO_TREND"


def _classify_volatility(atr_percent: float) -> str:
    """
    Classify volatility based on ATR percentage.
    
    Returns: "EXTREME", "HIGH", "NORMAL", "LOW"
    """
    if atr_percent >= 0.6:
        return "EXTREME"
    elif atr_percent >= 0.3:
        return "HIGH"
    elif atr_percent >= 0.1:
        return "NORMAL"
    else:
        return "LOW"


def _classify_momentum(rsi: float, macd_hist: float) -> str:
    """
    Classify momentum based on RSI and MACD.
    
    Returns: "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
    """
    if rsi > 70 and macd_hist > 0:
        return "STRONG_BULLISH"
    elif rsi > 50 and macd_hist > 0:
        return "BULLISH"
    elif rsi < 30 and macd_hist < 0:
        return "STRONG_BEARISH"
    elif rsi < 50 and macd_hist < 0:
        return "BEARISH"
    else:
        return "NEUTRAL"


def _analyze_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze market structure: swing highs/lows, higher-highs, lower-lows.
    """
    if len(df) < 40:
        return {"structure": "UNKNOWN", "higher_highs": False, "lower_lows": False}
    
    # Recent 20 candles vs previous 20 candles
    recent_high = df['high'].iloc[-20:].max()
    recent_low = df['low'].iloc[-20:].min()
    prev_high = df['high'].iloc[-40:-20].max()
    prev_low = df['low'].iloc[-40:-20].min()
    
    higher_highs = recent_high > prev_high
    lower_lows = recent_low < prev_low
    
    if higher_highs and not lower_lows:
        structure = "BULLISH"
    elif lower_lows and not higher_highs:
        structure = "BEARISH"
    elif higher_highs and lower_lows:
        structure = "EXPANDING"  # Volatility expansion
    else:
        structure = "RANGING"
    
    return {
        "structure": structure,
        "higher_highs": higher_highs,
        "lower_lows": lower_lows,
        "swing_high": float(recent_high),
        "swing_low": float(recent_low)
    }


def _detect_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect candlestick patterns.
    """
    if len(df) < 3:
        return {"patterns": [], "signal": None}
    
    patterns = []
    
    # Get last 3 candles
    c0 = df.iloc[-1]  # Current
    c1 = df.iloc[-2]  # Previous
    c2 = df.iloc[-3]  # 2 candles ago
    
    body0 = abs(c0['close'] - c0['open'])
    range0 = c0['high'] - c0['low']
    
    # === HAMMER ===
    if range0 > 0:
        lower_shadow = min(c0['open'], c0['close']) - c0['low']
        upper_shadow = c0['high'] - max(c0['open'], c0['close'])
        if lower_shadow > 2 * body0 and upper_shadow < body0 * 0.3 and body0 > 0:
            patterns.append("HAMMER")
    
    # === SHOOTING STAR ===
    if range0 > 0:
        lower_shadow = min(c0['open'], c0['close']) - c0['low']
        upper_shadow = c0['high'] - max(c0['open'], c0['close'])
        if upper_shadow > 2 * body0 and lower_shadow < body0 * 0.3 and body0 > 0:
            patterns.append("SHOOTING_STAR")
    
    # === BULLISH ENGULFING ===
    if c1['close'] < c1['open'] and c0['close'] > c0['open']:  # Prev bearish, current bullish
        if c0['open'] < c1['close'] and c0['close'] > c1['open']:
            patterns.append("BULLISH_ENGULFING")
    
    # === BEARISH ENGULFING ===
    if c1['close'] > c1['open'] and c0['close'] < c0['open']:  # Prev bullish, current bearish
        if c0['open'] > c1['close'] and c0['close'] < c1['open']:
            patterns.append("BEARISH_ENGULFING")
    
    # === DOJI ===
    if body0 < range0 * 0.1:
        patterns.append("DOJI")
    
    # === MORNING STAR (3-candle bullish reversal) ===
    body1 = abs(c1['close'] - c1['open'])
    body2 = abs(c2['close'] - c2['open'])
    if (c2['close'] < c2['open'] and  # First candle bearish
        body1 < body2 * 0.3 and  # Middle candle small
        c0['close'] > c0['open'] and  # Last candle bullish
        c0['close'] > (c2['open'] + c2['close']) / 2):  # Closes above midpoint
        patterns.append("MORNING_STAR")
    
    # === EVENING STAR (3-candle bearish reversal) ===
    if (c2['close'] > c2['open'] and  # First candle bullish
        body1 < body2 * 0.3 and  # Middle candle small
        c0['close'] < c0['open'] and  # Last candle bearish
        c0['close'] < (c2['open'] + c2['close']) / 2):  # Closes below midpoint
        patterns.append("EVENING_STAR")
    
    # Determine overall signal from patterns
    bullish_patterns = ["HAMMER", "BULLISH_ENGULFING", "MORNING_STAR"]
    bearish_patterns = ["SHOOTING_STAR", "BEARISH_ENGULFING", "EVENING_STAR"]
    
    signal = None
    if any(p in patterns for p in bullish_patterns):
        signal = "BULLISH"
    elif any(p in patterns for p in bearish_patterns):
        signal = "BEARISH"
    
    return {
        "patterns": patterns,
        "signal": signal
    }


def _calculate_risk_level(indicators: Dict[str, Any]) -> str:
    """
    Calculate overall risk level based on multiple factors.
    
    Returns: "EXTREME", "HIGH", "NORMAL", "LOW"
    """
    risk_score = 0
    
    # Volatility component
    volatility = indicators.get('volatility', 'NORMAL')
    if volatility == "EXTREME":
        risk_score += 40
    elif volatility == "HIGH":
        risk_score += 25
    elif volatility == "LOW":
        risk_score += 5
    else:
        risk_score += 10
    
    # ADX component (low ADX = choppy = risky)
    adx = indicators.get('adx14', 25)
    if adx < 15:
        risk_score += 30
    elif adx < 20:
        risk_score += 20
    elif adx < 25:
        risk_score += 10
    
    # RSI extremes
    rsi = indicators.get('rsi14', 50)
    if rsi > 85 or rsi < 15:
        risk_score += 20
    elif rsi > 75 or rsi < 25:
        risk_score += 10
    
    # BB position (price at extremes)
    bb_pos = indicators.get('bb_position', 0.5)
    if bb_pos > 0.95 or bb_pos < 0.05:
        risk_score += 15
    
    # Classification
    if risk_score >= 70:
        return "EXTREME"
    elif risk_score >= 50:
        return "HIGH"
    elif risk_score >= 25:
        return "NORMAL"
    else:
        return "LOW"


def _empty_indicators() -> Dict[str, Any]:
    """Return empty indicators when data is insufficient."""
    return {
        "ema9": 0, "ema21": 0, "ema50": 0,
        "sma20": 0, "sma50": 0, "sma200": None,
        "rsi14": 50, "rsi7": 50,
        "macd_line": 0, "macd_signal": 0, "macd_histogram": 0,
        "adx14": 25,
        "atr14": 0, "atr_percent": 0,
        "bb_upper": 0, "bb_middle": 0, "bb_lower": 0, "bb_width": 0, "bb_position": 0.5,
        "close": 0, "open": 0, "high": 0, "low": 0, "volume": 0,
        "trend": "UNKNOWN", "trend_strength": "UNKNOWN",
        "volatility": "UNKNOWN", "momentum": "UNKNOWN",
        "market_structure": {"structure": "UNKNOWN"},
        "patterns": {"patterns": [], "signal": None},
        "risk_level": "HIGH"
    }


def get_current_session() -> str:
    """
    Determine current trading session based on UTC time.
    Discrete sessions — no overlap window.
    
    Returns: "TOKYO", "LONDON", "NEW_YORK", "WEEKEND"
    """
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()
    
    if weekday >= 5:
        return "WEEKEND"
    
    # Discrete sessions (UTC) — no overlap
    # Tokyo:    21:00 - 07:00
    # London:   07:00 - 13:00
    # New York: 13:00 - 21:00
    
    if 7 <= hour < 13:
        return "LONDON"
    elif 13 <= hour < 21:
        return "NEW_YORK"
    else:
        return "TOKYO"


