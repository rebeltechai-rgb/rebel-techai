"""
REBEL Scanner - Market Scanner Module
Scans symbols from YAML configuration and generates trading signals.
Broker-agnostic via connector interface.
"""

import yaml
import os
import datetime
import time
from typing import List, Optional, TYPE_CHECKING
import MetaTrader5 as mt5

from rebel_signals import RebelSignals
from rebel_risk import RebelRisk

if TYPE_CHECKING:
    from connectors.base_connector import BrokerConnector

CONFIG_PATH = r"C:\Rebel Technologies\Rebel Master\Config\master_config.yaml"


def resolve_symbol_to_broker(config_symbol: str, all_mt5_symbols: list) -> Optional[str]:
    """
    Resolve a config symbol (e.g., AUDUSD) to the actual broker symbol.
    
    Priority:
    1. Exact match (AUDUSD)
    2. Standard suffixes (.a, .m, .fs, etc.) - NOT .sa (Select account)
    3. .sa suffix as last resort
    
    Args:
        config_symbol: Symbol name from config (e.g., "AUDUSD")
        all_mt5_symbols: List of MT5 symbol objects from mt5.symbols_get()
    
    Returns:
        Actual broker symbol name or None if not found
    """
    # Build lookup table of lowercase symbol names
    sym_map = {s.name.lower(): s.name for s in all_mt5_symbols}
    
    base = config_symbol.lower()
    
    # 1. Exact match first
    if base in sym_map:
        return sym_map[base]
    
    # 2. Find all matches that start with base
    matches = []
    for s in all_mt5_symbols:
        if s.name.lower().startswith(base):
            matches.append(s.name)
    
    if not matches:
        return None
    
    # 3. Prefer NON-.sa suffixes (standard account over Select)
    # Priority order: no suffix > .a > .m > .fs > others > .sa
    standard_matches = [m for m in matches if not m.lower().endswith('.sa')]
    
    if standard_matches:
        # Return shortest match (most likely the standard one)
        return min(standard_matches, key=len)
    
    # 4. Fall back to .sa if that's all we have
    return matches[0]


def extract_ml_features(symbol: str, raw_data: dict, indicators: dict, signal: dict) -> dict:
    """
    Extract ML features from scanner data for model training/inference.
    Produces a perfect ML training row with OHLC, candle anatomy, indicators,
    and signal metadata.
    
    Args:
        symbol: Trading symbol
        raw_data: Dict with 'entry', 'filter', 'trend' DataFrames (M5, M15, H1)
        indicators: Dict with ema9, ema21, ema50, rsi, atr, adx values
        signal: Signal dict with direction, score, breakdown
        
    Returns:
        Feature dictionary ready for ML pipeline
    """
    import pandas as pd
    
    now = pd.Timestamp.now(tz="UTC")
    
    features = {
        "symbol": symbol,
        "timestamp": now.isoformat(),
    }
    
    # -----------------------------------------------------------------
    # 1. OHLC & CANDLE ANATOMY (from entry timeframe M5)
    # -----------------------------------------------------------------
    df_entry = raw_data.get("entry")
    if df_entry is not None and len(df_entry) > 0:
        last = df_entry.iloc[-1]
        o = float(last["open"])
        h = float(last["high"])
        l = float(last["low"])
        c = float(last["close"])
        
        features["open"] = round(o, 6)
        features["high"] = round(h, 6)
        features["low"] = round(l, 6)
        features["close"] = round(c, 6)
        
        # Candle anatomy
        body = abs(c - o)
        total_range = h - l if h > l else 1e-10
        
        features["body"] = round(body, 6)
        features["wick_upper"] = round(h - max(o, c), 6)
        features["wick_lower"] = round(min(o, c) - l, 6)
        features["body_ratio"] = round(body / total_range, 4) if total_range > 0 else 0.0
        features["candle_direction"] = 1 if c > o else (-1 if c < o else 0)
    else:
        features["open"] = None
        features["high"] = None
        features["low"] = None
        features["close"] = None
        features["body"] = None
        features["wick_upper"] = None
        features["wick_lower"] = None
        features["body_ratio"] = None
        features["candle_direction"] = None
    
    # -----------------------------------------------------------------
    # 2. CORE INDICATOR VALUES
    # -----------------------------------------------------------------
    ema9 = indicators.get("ema9")
    ema21 = indicators.get("ema21")
    ema50 = indicators.get("ema50")
    rsi = indicators.get("rsi")
    atr = indicators.get("atr")
    adx = indicators.get("adx")
    
    features["atr"] = round(atr, 6) if atr else None
    features["rsi"] = round(rsi, 2) if rsi else None
    features["adx"] = round(adx, 2) if adx else None
    features["ema_fast"] = round(ema9, 6) if ema9 else None
    features["ema_slow"] = round(ema21, 6) if ema21 else None
    features["ema_trend"] = round(ema50, 6) if ema50 else None
    
    # MACD histogram (approximation: EMA9-EMA21 spread)
    if ema9 and ema21:
        features["macd_hist"] = round(ema9 - ema21, 6)
    else:
        features["macd_hist"] = None
    
    # -----------------------------------------------------------------
    # 3. TREND BIAS (derived from H1 EMA50 position)
    # -----------------------------------------------------------------
    close_price = features.get("close")
    if close_price and ema50:
        if close_price > ema50:
            features["trend_bias"] = "bullish"
        elif close_price < ema50:
            features["trend_bias"] = "bearish"
        else:
            features["trend_bias"] = "neutral"
    else:
        features["trend_bias"] = signal.get("trend_bias", "unknown")
    
    # -----------------------------------------------------------------
    # 4. VOLATILITY REGIME (ATR vs median ATR)
    # -----------------------------------------------------------------
    df_filter = raw_data.get("filter")
    if df_filter is not None and len(df_filter) >= 50 and atr:
        # Calculate ATR series for filter timeframe
        high_f = df_filter["high"]
        low_f = df_filter["low"]
        close_f = df_filter["close"]
        prev_close = close_f.shift(1)
        tr = pd.concat([
            high_f - low_f,
            (high_f - prev_close).abs(),
            (low_f - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean().dropna()
        
        if len(atr_series) > 20:
            atr_median = float(atr_series.tail(100).median())
            if atr_median > 0:
                atr_ratio = atr / atr_median
                if atr_ratio > 1.5:
                    features["volatility_regime"] = "expanding"
                elif atr_ratio < 0.7:
                    features["volatility_regime"] = "contracting"
                else:
                    features["volatility_regime"] = "normal"
            else:
                features["volatility_regime"] = "unknown"
        else:
            features["volatility_regime"] = "unknown"
    else:
        features["volatility_regime"] = "unknown"
    
    # -----------------------------------------------------------------
    # 5. SESSION STATE (based on UTC hour)
    # -----------------------------------------------------------------
    hour = now.hour
    if 7 <= hour < 16:  # London session (approx)
        features["session_state"] = "london"
    elif 13 <= hour < 22:  # NY session overlaps
        features["session_state"] = "newyork"
    elif 0 <= hour < 9:  # Asian session
        features["session_state"] = "asian"
    else:
        features["session_state"] = "off_hours"
    
    # -----------------------------------------------------------------
    # 6. SIGNAL METADATA
    # -----------------------------------------------------------------
    features["signal_score"] = signal.get("score", 0)
    
    direction = signal.get("direction")
    features["raw_signal"] = 1 if direction == "long" else (-1 if direction == "short" else 0)
    
    # Reason string (from breakdown)
    breakdown = signal.get("breakdown", {})
    reasons = []
    if breakdown.get("ema_crossover"):
        reasons.append("ema_cross")
    if breakdown.get("trend_filter"):
        reasons.append("trend_aligned")
    if breakdown.get("rsi_agrees"):
        reasons.append("rsi_confirms")
    if breakdown.get("atr_valid"):
        reasons.append("atr_ok")
    if breakdown.get("adx_valid"):
        reasons.append("adx_strong")
    
    features["reason"] = ",".join(reasons) if reasons else "none"
    
    # -----------------------------------------------------------------
    # 7. ADDITIONAL DERIVED FEATURES
    # -----------------------------------------------------------------
    # EMA spreads normalized by ATR
    if atr and atr > 0:
        if ema9 and ema21:
            features["ema_fast_slow_spread_atr"] = round((ema9 - ema21) / atr, 4)
        else:
            features["ema_fast_slow_spread_atr"] = None
        
        if close_price and ema50:
            features["price_ema50_dist_atr"] = round((close_price - ema50) / atr, 4)
        else:
            features["price_ema50_dist_atr"] = None
    else:
        features["ema_fast_slow_spread_atr"] = None
        features["price_ema50_dist_atr"] = None
    
    # Recent returns (for momentum features)
    if df_entry is not None and len(df_entry) >= 21:
        close_series = df_entry["close"]
        c1 = float(close_series.iloc[-2])
        c5 = float(close_series.iloc[-6])
        c20 = float(close_series.iloc[-21])
        c_now = float(close_series.iloc[-1])
        
        features["return_1bar_pct"] = round((c_now - c1) / c1 * 100, 4) if c1 != 0 else None
        features["return_5bar_pct"] = round((c_now - c5) / c5 * 100, 4) if c5 != 0 else None
        features["return_20bar_pct"] = round((c_now - c20) / c20 * 100, 4) if c20 != 0 else None
    else:
        features["return_1bar_pct"] = None
        features["return_5bar_pct"] = None
        features["return_20bar_pct"] = None
    
    return features


class RebelScanner:
    """Market scanner for the REBEL trading bot."""
    
    def __init__(self, broker: "BrokerConnector", config: dict = None):
        """
        Initialize the scanner with a broker connector and configuration.
        
        Args:
            broker: BrokerConnector instance for market data
            config: Master configuration dictionary
        """
        self.broker = broker
        self.config = config or self._load_master_config()
        self.symbols = []
        self.signals = RebelSignals(broker)
        self.risk = RebelRisk(broker, self.config.get("risk", {}))
        
        # Scanner settings
        scanner_config = self.config.get("scanner", {})
        self.enabled = scanner_config.get("enabled", True)
        self.timeframe = scanner_config.get("timeframe", "M15")
        self.max_symbols = scanner_config.get("max_symbols", 25)
        self.use_symbol_groups = scanner_config.get("use_symbol_groups", True)
        self.debug_signals = scanner_config.get("debug_signals", False)

        # Optional blocklist from engine (symbols to skip scanning)
        self.blocklist = set()
        
        # Symbol count tracking for dashboard
        self.last_total_symbols = 0
        self.last_active_symbols = 0
        
        # Symbol resolution flag
        self._symbols_resolved = False
        
        # Load symbols from config (resolution happens later when MT5 is connected)
        self._load_config_symbols()
    
    def _load_master_config(self) -> dict:
        """Load master configuration file."""
        if not os.path.exists(CONFIG_PATH):
            return {}
        
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    
    def _load_config_symbols(self) -> List[str]:
        """
        Load symbols from config WITHOUT resolution (MT5 may not be connected yet).
        """
        cfg_symbols = self.config.get("symbols", {})
        groups = cfg_symbols.get("groups", {})

        config_symbols = []

        # Load all symbols from all groups
        for group_name, group_cfg in groups.items():
            if not group_cfg.get("enabled", True):
                continue

            group_symbols = group_cfg.get("symbols", [])
            if isinstance(group_symbols, list):
                config_symbols.extend(group_symbols)

        # Remove duplicates
        self.symbols = list(dict.fromkeys(config_symbols))
        self._config_symbols = self.symbols.copy()  # Keep original for resolution
        
        return self.symbols
    
    def resolve_symbols(self) -> List[str]:
        """
        Resolve config symbols to actual broker symbols (call after MT5 connected).
        Handles broker-specific suffixes (e.g., AUDUSD → AUDUSD.sa).
        """
        if self._symbols_resolved:
            return self.symbols
        
        all_mt5_symbols = None
        for attempt in range(5):
            all_mt5_symbols = mt5.symbols_get()
            if all_mt5_symbols:
                break
            print(f"[SCANNER] WARNING: MT5 symbols not available (attempt {attempt + 1}/5)")
            time.sleep(1)
        if not all_mt5_symbols:
            print("[SCANNER] WARNING: Could not get MT5 symbols, using config names")
            return self.symbols
        
        self.symbols = []
        resolved_count = 0
        
        for config_sym in self._config_symbols:
            broker_sym = resolve_symbol_to_broker(config_sym, all_mt5_symbols)
            
            if broker_sym:
                self.symbols.append(broker_sym)
                if broker_sym != config_sym:
                    resolved_count += 1
            else:
                print(f"[SCANNER] WARNING: {config_sym} not found on broker, skipping")
        
        if resolved_count > 0:
            print(f"[SCANNER] Resolved {resolved_count} symbols to broker format (e.g., AUDUSD → AUDUSD.sa)")
        
        # If user chose full scanning (mode B), ignore max_symbols
        if not self.use_symbol_groups and len(self.symbols) > self.max_symbols:
            self.symbols = self.symbols[:self.max_symbols]
        
        print(f"[SCANNER] Loaded {len(self.symbols)} tradeable symbols")
        self._symbols_resolved = True
        
        return self.symbols

    def set_symbol_blocklist(self, symbols: List[str]) -> None:
        """Set symbols to skip during scanning."""
        self.blocklist = set(symbols or [])
    
    def load_symbols(self) -> List[str]:
        """Legacy method - calls resolve_symbols."""
        return self.resolve_symbols()

    ###############################################################
    # MARKET HOURS INTELLIGENCE (QUIET MODE)
    ###############################################################
    def market_is_open(self, symbol: str) -> bool:
        """Returns True if market session is open (UTC)."""

        now = datetime.datetime.now(datetime.timezone.utc)
        weekday = now.strftime("%a").lower()[:3]   # mon, tue, wed...

        cfg_sessions = self.config.get("sessions", {})

        # -------- Determine symbol category ----------
        session_key = None

        groups = self.config.get("symbols", {}).get("groups", {})

        if groups.get("fx_majors") and symbol in groups["fx_majors"].get("symbols", []):
            session_key = "fx"
        elif groups.get("fx_crosses") and symbol in groups["fx_crosses"].get("symbols", []):
            session_key = "fx"
        elif groups.get("fx_exotics") and symbol in groups["fx_exotics"].get("symbols", []):
            session_key = "fx"
        elif groups.get("metals") and symbol in groups["metals"].get("symbols", []):
            session_key = "metals"
        elif groups.get("energies") and symbol in groups["energies"].get("symbols", []):
            session_key = "energies"
        elif groups.get("softs_commodities") and symbol in groups["softs_commodities"].get("symbols", []):
            session_key = "softs"
        elif groups.get("crypto") and symbol in groups["crypto"].get("symbols", []):
            session_key = "crypto"
        elif groups.get("indices") and symbol in groups["indices"].get("symbols", []):
            # auto-region recognition
            name = symbol.upper()
            if any(tag in name for tag in ["DAX", "FRA", "CAC", "UK", "EU", "SPA", "SWI", "IT4", "NETH"]):
                session_key = "indices_eu"
            elif any(tag in name for tag in ["NAS", "UST", "US5", "DJ", "DOW"]):
                session_key = "indices_us"
            elif any(tag in name for tag in ["JP", "NK", "225", "HK", "CHINA"]):
                session_key = "indices_asia"

        # Unknown symbol → assume always open
        if session_key is None:
            return True

        session_cfg = cfg_sessions.get(session_key)
        if session_cfg is None:
            return True

        # Check weekday
        if weekday not in session_cfg.get("days", []):
            return False

        # Open/close times
        open_str = session_cfg.get("open", "00:00")
        close_str = session_cfg.get("close", "23:59")

        open_hour, open_min = map(int, open_str.split(":"))
        close_hour, close_min = map(int, close_str.split(":"))

        open_dt = now.replace(hour=open_hour, minute=open_min, second=0, microsecond=0)
        close_dt = now.replace(hour=close_hour, minute=close_min, second=0, microsecond=0)

        return open_dt <= now <= close_dt

    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is available via the broker.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            True if symbol is available
        """
        return self.broker.ensure_symbol(symbol)
    
    def get_valid_symbols(self) -> List[str]:
        """
        Get list of valid, tradeable symbols.
        
        Returns:
            List of valid symbol strings
        """
        valid = []
        for symbol in self.symbols:
            if self.validate_symbol(symbol):
                valid.append(symbol)
            else:
                print(f"[SCANNER] Symbol not available: {symbol}")
        
        return valid
    
    def _format_indicator(self, value, decimals: int = 5) -> str:
        """Format indicator value for display."""
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}"
    
    # ==========================================================
    # ML FEATURE EXTRACTION (Step 6 – Option B)
    # Produces ML-ready feature dictionary for the engine
    # MUST match MLFeatureLogger columns EXACTLY
    # ==========================================================
    def extract_ml_features(self, symbol, data, indicators, signal):
        """
        Create a dictionary of ML features for the engine to log.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with OHLC data (entry timeframe)
            indicators: Dict with indicator values
            signal: Signal dict with direction, score, etc.
            
        Returns:
            Feature dictionary matching MLFeatureLogger columns EXACTLY
        """
        try:
            # Get ATR value (safe)
            atr = float(indicators.get("atr", 0.0))
            
            # -------- MACD HISTOGRAM --------
            try:
                close_prices = data["close"]
                ema12 = close_prices.ewm(span=12, adjust=False).mean()
                ema26 = close_prices.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                macd_hist = float(macd_line.iloc[-1] - signal_line.iloc[-1])
            except Exception:
                macd_hist = 0.0
            
            # -------- SPREAD RATIO --------
            try:
                # Try to get spread from broker
                tick = self.broker.get_symbol_tick(symbol) if hasattr(self.broker, 'get_symbol_tick') else None
                if tick:
                    spread = tick.get("spread", 0) if isinstance(tick, dict) else getattr(tick, 'spread', 0)
                else:
                    # Fallback: try MT5 directly
                    import MetaTrader5 as mt5
                    tick_info = mt5.symbol_info_tick(symbol)
                    if tick_info:
                        spread = tick_info.ask - tick_info.bid
                    else:
                        spread = 0
                spread_ratio = spread / atr if atr > 0 else 0
            except Exception:
                spread_ratio = 0.0
            
            # Volatility regime calculation
            if data is not None and len(data) >= 30:
                # Calculate ATR for recent bars if not in data
                if "atr" in data.columns:
                    atr_30 = float(data["atr"].tail(30).mean())
                else:
                    # Calculate simple ATR proxy from range
                    recent_ranges = (data["high"] - data["low"]).tail(30)
                    atr_30 = float(recent_ranges.mean())
                
                if atr_30 > 0:
                    volatility_regime = round(atr / atr_30, 4)
                else:
                    volatility_regime = 1.0
            else:
                volatility_regime = 1.0
            
            # Session state (0-Asia, 1-London, 2-NY, 3-Rollover)
            hour = datetime.datetime.now(datetime.timezone.utc).hour
            if 0 <= hour < 7:
                session_state = 0
            elif 7 <= hour < 13:
                session_state = 1
            elif 13 <= hour < 20:
                session_state = 2
            else:
                session_state = 3
            
            # Determine trend bias
            trend_bias = signal.get("trend_bias", "neutral")
            if trend_bias == "bullish":
                trend_bias_val = 1
            elif trend_bias == "bearish":
                trend_bias_val = -1
            else:
                trend_bias_val = 0
            
            # Raw signal encoding
            direction = signal.get("direction")
            if direction == "long":
                raw_signal = 1
            elif direction == "short":
                raw_signal = -1
            else:
                raw_signal = 0
            
            # Build reason string from breakdown (with fallback)
            breakdown = signal.get("breakdown", {})
            reasons = []
            if breakdown.get("ema_crossover"):
                reasons.append("ema_cross")
            if breakdown.get("trend_filter"):
                reasons.append("trend_aligned")
            if breakdown.get("rsi_agrees"):
                reasons.append("rsi_confirms")
            if breakdown.get("atr_valid"):
                reasons.append("atr_ok")
            if breakdown.get("adx_valid"):
                reasons.append("adx_strong")
            
            # Reason with proper fallback - never blank
            if reasons:
                reason_str = ",".join(reasons)
            else:
                reason_str = signal.get("reason") or signal.get("trend_bias") or "none"
            
            # Construct ML feature dict EXACTLY matching MLFeatureLogger columns
            return {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "ema_fast": indicators.get("ema9"),
                "ema_slow": indicators.get("ema21"),
                "rsi": indicators.get("rsi"),
                "atr": atr,
                "adx": indicators.get("adx"),
                "macd_hist": round(macd_hist, 6),
                "trend_bias": trend_bias_val,
                "volatility_regime": volatility_regime,
                "spread_ratio": round(spread_ratio, 6),
                "session_state": session_state,
                "raw_signal": raw_signal,
                "signal_score": signal.get("score", 0),
                "reason": reason_str
            }

        except Exception as e:
            print(f"[ML_FEATURES] WARNING: failed for {symbol}: {e}")
            return {}
    
    def _print_debug_output(self, signal: dict):
        """
        Print detailed debug output for a signal.
        
        Args:
            signal: Signal dictionary from RebelSignals
        """
        symbol = signal.get("symbol", "UNKNOWN")
        indicators = signal.get("indicators", {})
        breakdown = signal.get("breakdown", {})
        score = signal.get("score", 0)
        direction = signal.get("direction")
        
        print(f"\n[DEBUG] {symbol}")
        
        # Indicator values
        ema9 = self._format_indicator(indicators.get("ema9"), 5)
        ema21 = self._format_indicator(indicators.get("ema21"), 5)
        ema50 = self._format_indicator(indicators.get("ema50"), 5)
        print(f"  ema9: {ema9}  ema21: {ema21}  ema50: {ema50}")
        
        rsi = self._format_indicator(indicators.get("rsi"), 1)
        atr = self._format_indicator(indicators.get("atr"), 5)
        adx = self._format_indicator(indicators.get("adx"), 1)
        print(f"  rsi: {rsi}  atr: {atr}  adx: {adx}")
        
        # Breakdown
        print("  breakdown:")
        print(f"     ema_crossover: {breakdown.get('ema_crossover', False)}")
        print(f"     trend_filter: {breakdown.get('trend_filter', False)}")
        print(f"     rsi_agrees: {breakdown.get('rsi_agrees', False)}")
        print(f"     atr_valid: {breakdown.get('atr_valid', False)}")
        print(f"     adx_valid: {breakdown.get('adx_valid', False)}")
        
        # Final result
        dir_str = direction if direction else "none"
        score_100 = int(round((score / 5.0) * 100)) if isinstance(score, (int, float)) else 0
        print(f"  => score: {score} ({score_100}/100) | direction: {dir_str}")
    
    def scan_symbol(self, symbol: str, include_features: bool = True) -> Optional[dict]:
        """
        Scan a single symbol for trading signals.
        
        Args:
            symbol: Symbol to scan
            include_features: Whether to include ML features in output
            
        Returns:
            Signal dictionary with optional features key, or None
        """
        # Validate symbol
        if not self.validate_symbol(symbol):
            return None
        
        # Generate signal
        signal = self.signals.generate_signal(symbol, self.timeframe)
        
        # Print debug output if enabled
        if self.debug_signals:
            self._print_debug_output(signal)
        
        # Only return if there's an actual signal (direction != None means score >= 3)
        if signal.get("direction") is not None:
            score = signal.get("score", 0)
            signal["score_100"] = int(round((score / 5.0) * 100)) if isinstance(score, (int, float)) else 0
            # Add trend_bias to signal for reference
            indicators = signal.get("indicators", {})
            ema50 = indicators.get("ema50")
            close_price = None
            
            # Fetch raw data for ML features
            df = self.signals.fetch_ohlc(symbol, self.signals.ENTRY_TF, self.signals.BARS_COUNT)
            
            if df is not None and len(df) > 0:
                close_price = float(df["close"].iloc[-1])
            
            # Determine trend bias
            if close_price and ema50:
                if close_price > ema50:
                    signal["trend_bias"] = "bullish"
                elif close_price < ema50:
                    signal["trend_bias"] = "bearish"
                else:
                    signal["trend_bias"] = "neutral"
            else:
                signal["trend_bias"] = "unknown"
            
            # Extract ML features (ENGINE will log them)
            if include_features and df is not None:
                signal["features"] = self.extract_ml_features(
                    symbol=symbol,
                    data=df,
                    indicators=indicators,
                    signal=signal
                )
            
            return signal
        
        return None
    
    def scan(self) -> List[dict]:
        """
        Scan all symbols for trading signals.
        
        Returns:
            List of signal dictionaries
        """
        if not self.enabled:
            print("[SCANNER] Scanner is disabled")
            return []
        
        # Resolve symbols on first scan (MT5 should be connected by now)
        if not self._symbols_resolved:
            self.resolve_symbols()
        
        signals = []
        valid_symbols = self.get_valid_symbols()
        blocked_count = 0
        if self.blocklist:
            filtered = [s for s in valid_symbols if s not in self.blocklist]
            blocked_count = len(valid_symbols) - len(filtered)
            valid_symbols = filtered
        
        # Track symbol counts for dashboard
        total_symbols = len(valid_symbols)
        active_symbols = 0
        
        print(f"[SCANNER] Scanning {total_symbols} symbols on {self.timeframe}...")
        if blocked_count > 0:
            print(f"[SCANNER] Skipping {blocked_count} symbols (never passed filters)")
        
        if self.debug_signals:
            print("[SCANNER] Debug mode enabled - showing all indicator values")
        
        for symbol in valid_symbols:

            # ----------------------------------------------------
            # QUIET MODE — SKIP CLOSED MARKETS WITHOUT LOGGING
            # ----------------------------------------------------
            if not self.market_is_open(symbol):
                continue

            # Symbol passed market hours check - count as active
            active_symbols += 1

            signal = self.scan_symbol(symbol)
            
            if signal is not None:
                # Run risk validation
                risk_check = self.risk.validate_trade(symbol)
                signal["risk_valid"] = risk_check["can_trade"]
                signal["risk_messages"] = risk_check["messages"]
                
                signals.append(signal)
                
                if not self.debug_signals:
                    # Only print summary if not in debug mode (debug already printed details)
                    print(f"[SCANNER] Signal: {symbol} -> {signal['direction'].upper()} (score: {signal['score']}/5)")
        
        # Store counts for dashboard access
        self.last_total_symbols = total_symbols
        self.last_active_symbols = active_symbols
        
        print(f"\n[SCANNER] Scan complete. Found {len(signals)} signals.")
        return signals
    
    def get_scanner_status(self) -> dict:
        """
        Get current scanner status.
        
        Returns:
            Status dictionary
        """
        return {
            "enabled": self.enabled,
            "timeframe": self.timeframe,
            "total_symbols": self.last_total_symbols if self.last_total_symbols > 0 else len(self.symbols),
            "active_symbols": self.last_active_symbols,
            "max_symbols": self.max_symbols,
            "debug_signals": self.debug_signals,
            "symbols": self.symbols
        }
