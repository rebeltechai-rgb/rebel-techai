"""
REBEL Signals - Multi-Timeframe Indicator and Signal Generation Module
Uses EMA, RSI, ATR, ADX across M5/M15/H1 timeframes with scoring system.
Pure pandas implementation - broker-agnostic via connector interface.
"""

import pandas as pd
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from connectors.base_connector import BrokerConnector


class RebelSignals:
    """Multi-timeframe technical indicator and signal generation class."""
    
    # Timeframe configuration (single timeframe: H1 only)
    ENTRY_TF = "H1"
    FILTER_TF = "H1"
    TREND_TF = "H1"
    BARS_COUNT = 500
    
    # Indicator parameters
    EMA_FAST = 9
    EMA_MID = 21
    EMA_SLOW = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    
    # Thresholds (RELAXED FOR ML DATA COLLECTION)
    RSI_LONG_MIN = 55       # Reverted for RF training - need diverse data
    RSI_SHORT_MAX = 45      # Reverted for RF training - need diverse data
    ADX_MIN = 25            # Default - overridden per asset class below
    ATR_LOW_MULT = 0.5
    ATR_HIGH_MULT = 2.0
    MIN_SCORE = 4           # Was 3 - require higher quality signals
    MIN_BARS = 100
    
    # Asset-class specific ADX thresholds
    # Lowered for underrepresented classes to improve diversity
    ADX_THRESHOLDS = {
        "fx_major": 20,     # EURUSD, GBPUSD, USDJPY etc - frozen baseline v1.3
        "fx_minor": 22,     # Crosses - frozen baseline v1.3
        "fx_exotic": 25,    # Exotics - stricter than minors
        "metals": 30,       # Metals - stronger trend filter
        "crypto": 30,       # BTC/ETH etc - keep strict (already over-trading)
        "indices": 30,      # Indices - stronger trend filter
        "energies": 30,     # Energies - stronger trend filter
        "softs": 30,        # Softs - stronger trend filter
        "default": 25       # Fallback
    }
    
    # Asset-class specific MIN_SCORE thresholds
    # Lower = more trades, Higher = fewer but higher quality
    # Based on historical win rates: Metals 100%, Indices 85%, Crypto 49%, FX 30%
    SCORE_THRESHOLDS = {
        "metals": 4,        # Tighter score gate
        "energies": 4,
        "softs": 4,
        "indices": 4,
        "crypto": 4,
        "fx_major": 4,
        "fx_minor": 4,
        "fx_exotic": 4,
        "default": 4
    }
    
    # FX Major pairs (subtler trends, lower ADX threshold)
    FX_MAJORS = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"}
    
    # FX Minor/Crosses
    FX_MINORS = {
        "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
        "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
        "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
        "NZDJPY", "NZDCAD", "NZDCHF", "CADJPY", "CADCHF", "CHFJPY"
    }
    
    def __init__(self, broker: "BrokerConnector", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the signals module with a broker connector.
        
        Args:
            broker: BrokerConnector instance for data fetching
        """
        self.broker = broker
        self.config = config or {}
        self.last_signals = {}
    
    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class for ADX/score threshold selection."""
        sym = symbol.upper()
        
        # Check FX majors first
        if sym in self.FX_MAJORS:
            return "fx_major"
        
        # Check FX minors
        if sym in self.FX_MINORS:
            return "fx_minor"
        
        # Crypto detection
        crypto_prefixes = ("BTC", "ETH", "LTC", "XRP", "ADA", "DOG", "DOT", "SOL",
                          "AVAX", "LNK", "UNI", "BCH", "XLM", "AAVE", "SUSHI")
        if any(c in sym for c in crypto_prefixes) or sym.endswith(("-USD", "-JPY")):
            return "crypto"
        
        # Metals detection (Gold, Silver, Platinum)
        if sym.startswith(("XAU", "XAG", "XPT", "GOLD", "SILVER", "COPPER")):
            return "metals"
        
        # Energies detection (Oil, Gas)
        energy_keywords = ("BRENT", "WTI", "OIL", "UKOIL", "USOIL", "CRUDE", "NATGAS", "GAS")
        if any(k in sym for k in energy_keywords):
            return "energies"
        
        # Softs/Agricultural detection
        soft_keywords = ("COCOA", "COFFEE", "SUGAR", "COTTON", "WHEAT", "CORN", "SOY", "BEAN", "SOYBEAN")
        if any(k in sym for k in soft_keywords):
            return "softs"
        
        # Indices detection
        idx_keywords = ("US500", "US30", "NAS", "DAX", "UK100", "GER40", "SPA35",
                       "AUS200", "JPN225", "HK50", "US2000", "USTECH", "DJ30",
                       "FT100", "S&P", "HSI", "NK225", "CAC40", "CHINA50", "EU50",
                       "FRA40", "IT40", "SWI20", "NETH25", "CN50", "VIX", "SGFREE",
                       "SPI200", "EUSTX50", "USDINDEX")
        if any(k in sym for k in idx_keywords):
            return "indices"
        
        # FX exotic detection (remaining currency pairs)
        exotic_currencies = ("ZAR", "MXN", "TRY", "PLN", "HUF", "CZK", "RON",
                            "IDR", "THB", "INR", "BRL", "CLP", "COP", "KRW", "TWD")
        if any(c in sym for c in exotic_currencies):
            return "fx_exotic"
        
        return "default"
    
    def _get_adx_threshold(self, symbol: str) -> float:
        """Get ADX threshold for a symbol based on its asset class."""
        sym_upper = symbol.upper()
        if "COCOA" in sym_upper:
            return 25
        if sym_upper.startswith(("XAU", "XAG", "XPT", "XPD")):
            return 30
        if any(k in sym_upper for k in ("BRENT", "WTI", "OIL", "UKOIL", "USOIL", "CRUDE", "NATGAS", "GAS")):
            return 30
        asset_class = self._get_asset_class(symbol)
        if asset_class == "crypto":
            return 30
        return self.ADX_THRESHOLDS.get(asset_class, self.ADX_THRESHOLDS["default"])
    
    def _get_score_threshold(self, symbol: str) -> int:
        """Get minimum score threshold for a symbol based on its asset class."""
        asset_class = self._get_asset_class(symbol)
        threshold = int(self.SCORE_THRESHOLDS.get(asset_class, self.SCORE_THRESHOLDS["default"]))
        strategy_cfg = self.config.get("strategy", {}) or {}
        mode = strategy_cfg.get("mode", "normal")
        min_score_map = strategy_cfg.get("min_score", {}) or {}
        cfg_threshold = min_score_map.get(mode)
        if isinstance(cfg_threshold, (int, float)):
            return max(int(cfg_threshold), threshold)
        return threshold

    def _get_indicator_overrides(self, symbol: str) -> Dict[str, Any]:
        overrides = (self.config.get("indicator_overrides") or {})
        if not overrides or not overrides.get("enabled", False):
            return {}
        defaults = overrides.get("defaults", {}) or {}
        families = overrides.get("families", {}) or {}
        family = self._get_asset_class(symbol)
        family_cfg = families.get(family, {}) or {}
        return {**defaults, **family_cfg}
    
    def fetch_ohlc(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch OHLC data via the broker connector.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe string (e.g., 'M5', 'M15', 'H1')
            bars: Number of bars to fetch (default: 500)
            
        Returns:
            DataFrame with OHLC data or None if failed
        """
        df = self.broker.get_historical_data(symbol, timeframe, bars)
        
        if df is None or len(df) < self.MIN_BARS:
            return None
        
        return df
    
    # =========================================================================
    # INDICATOR CALCULATIONS (Pure Pandas)
    # =========================================================================
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            series: Price series
            period: EMA period
            
        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI using Wilder's smoothing method.
        
        Args:
            close: Close price series
            period: RSI period (default: 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = close.diff()
        
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Wilder's smoothing (alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = TR.rolling(period).mean()
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default: 14)
            
        Returns:
            ATR series
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate ADX (Average Directional Index).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ADX period (default: 14)
            
        Returns:
            ADX series
        """
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        # +DM: up_move > down_move and up_move > 0
        plus_dm = pd.Series(0.0, index=high.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        
        # -DM: down_move > up_move and down_move > 0
        minus_dm = pd.Series(0.0, index=high.index)
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing
        atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # Calculate +DI and -DI
        di_plus = 100 * (smooth_plus_dm / atr.replace(0, 1e-10))
        di_minus = 100 * (smooth_minus_dm / atr.replace(0, 1e-10))
        
        # Calculate DX
        di_sum = di_plus + di_minus
        di_diff = (di_plus - di_minus).abs()
        dx = 100 * (di_diff / di_sum.replace(0, 1e-10))
        
        # Calculate ADX (smoothed DX)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    # =========================================================================
    # MULTI-TIMEFRAME SIGNAL GENERATION
    # =========================================================================
    
    def generate_signal(self, symbol: str, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Generate trading signal for a symbol using multi-timeframe analysis.
        
        Note: The timeframe parameter is ignored internally. We use hardcoded:
            - Entry: H1
            - Filter: H1
            - Trend: H1
        
        Args:
            symbol: Trading symbol
            timeframe: Ignored - kept for API compatibility
            
        Returns:
            Signal dictionary with direction, score, and indicator values
        """
        # Initialize neutral result
        result = {
            "symbol": symbol,
            "direction": None,
            "score": 0,
            "indicators": {
                "ema9": None,
                "ema21": None,
                "ema50": None,
                "rsi": None,
                "atr": None,
                "adx": None
            },
            "breakdown": {
                "ema_crossover": False,
                "trend_filter": False,
                "rsi_agrees": False,
                "atr_valid": False,
                "adx_valid": False
            }
        }
        
        # ---------------------------------------------------------------------
        # FETCH DATA FROM ALL THREE TIMEFRAMES
        # ---------------------------------------------------------------------
        df_entry = self.fetch_ohlc(symbol, self.ENTRY_TF, self.BARS_COUNT)
        df_filter = self.fetch_ohlc(symbol, self.FILTER_TF, self.BARS_COUNT)
        df_trend = self.fetch_ohlc(symbol, self.TREND_TF, self.BARS_COUNT)
        
        # Check if all data is available
        if df_entry is None or df_filter is None or df_trend is None:
            self.last_signals[symbol] = result
            return result
        
        # ---------------------------------------------------------------------
        # CALCULATE INDICATORS ON EACH TIMEFRAME
        # ---------------------------------------------------------------------
        
        overrides = self._get_indicator_overrides(symbol)
        ema_fast = int(overrides.get("ema_fast", self.EMA_FAST))
        ema_mid = int(overrides.get("ema_mid", self.EMA_MID))
        ema_slow = int(overrides.get("ema_slow", self.EMA_SLOW))
        rsi_period = int(overrides.get("rsi_period", self.RSI_PERIOD))
        rsi_long_min = float(overrides.get("rsi_long_min", self.RSI_LONG_MIN))
        rsi_short_max = float(overrides.get("rsi_short_max", self.RSI_SHORT_MAX))

        # TREND TIMEFRAME (H1): EMA50, ADX
        ema50_trend = self.calculate_ema(df_trend['close'], ema_slow)
        adx_trend = self.calculate_adx(df_trend['high'], df_trend['low'], df_trend['close'], self.ADX_PERIOD)
        
        ema50_trend_last = ema50_trend.iloc[-1]
        adx_trend_last = adx_trend.iloc[-1]
        close_trend_last = df_trend['close'].iloc[-1]
        
        # Determine trend direction from H1
        trend_direction = None
        if pd.notna(ema50_trend_last):
            if close_trend_last > ema50_trend_last:
                trend_direction = "long"
            elif close_trend_last < ema50_trend_last:
                trend_direction = "short"
        
        # FILTER TIMEFRAME (M15): RSI, ATR
        rsi_filter = self.calculate_rsi(df_filter['close'], rsi_period)
        atr_filter = self.calculate_atr(df_filter['high'], df_filter['low'], df_filter['close'], self.ATR_PERIOD)
        
        rsi_filter_last = rsi_filter.iloc[-1]
        atr_filter_last = atr_filter.iloc[-1]
        
        # Compute median ATR over last ~100 bars
        atr_for_median = atr_filter.dropna().tail(100)
        median_atr_filter = atr_for_median.median() if len(atr_for_median) > 0 else 0
        
        # ENTRY TIMEFRAME (M5): EMA9, EMA21
        ema9_entry = self.calculate_ema(df_entry['close'], ema_fast)
        ema21_entry = self.calculate_ema(df_entry['close'], ema_mid)
        
        ema9_entry_last = ema9_entry.iloc[-1]
        ema9_entry_prev = ema9_entry.iloc[-2]
        ema21_entry_last = ema21_entry.iloc[-1]
        ema21_entry_prev = ema21_entry.iloc[-2]
        
        # ---------------------------------------------------------------------
        # POPULATE INDICATOR VALUES IN RESULT
        # ---------------------------------------------------------------------
        result["indicators"] = {
            "ema9": round(ema9_entry_last, 5) if pd.notna(ema9_entry_last) else None,
            "ema21": round(ema21_entry_last, 5) if pd.notna(ema21_entry_last) else None,
            "ema50": round(ema50_trend_last, 5) if pd.notna(ema50_trend_last) else None,
            "rsi": round(rsi_filter_last, 2) if pd.notna(rsi_filter_last) else None,
            "atr": round(atr_filter_last, 5) if pd.notna(atr_filter_last) else None,
            "adx": round(adx_trend_last, 2) if pd.notna(adx_trend_last) else None
        }
        
        # ---------------------------------------------------------------------
        # DETECT EMA CROSSOVER ON M5
        # ---------------------------------------------------------------------
        long_cross = False
        short_cross = False
        
        if pd.notna(ema9_entry_last) and pd.notna(ema21_entry_last):
            if pd.notna(ema9_entry_prev) and pd.notna(ema21_entry_prev):
                # Bullish crossover: EMA9 crosses above EMA21
                long_cross = (ema9_entry_prev <= ema21_entry_prev) and (ema9_entry_last > ema21_entry_last)
                # Bearish crossover: EMA9 crosses below EMA21
                short_cross = (ema9_entry_prev >= ema21_entry_prev) and (ema9_entry_last < ema21_entry_last)
        
        # ---------------------------------------------------------------------
        # SCORING LOGIC
        # ---------------------------------------------------------------------
        score = 0
        direction = None
        breakdown = {
            "ema_crossover": False,
            "trend_filter": False,
            "rsi_agrees": False,
            "atr_valid": False,
            "adx_valid": False
        }
        
        # a) EMA Crossover (M5): +1
        candidate_direction = None
        if long_cross:
            candidate_direction = "long"
        elif short_cross:
            candidate_direction = "short"
        
        if candidate_direction is not None:
            breakdown["ema_crossover"] = True
            score += 1
            direction = candidate_direction
        
        # b) Trend Filter (H1 EMA50): +1
        if direction is not None and trend_direction is not None:
            if direction == "long" and trend_direction == "long":
                breakdown["trend_filter"] = True
                score += 1
            elif direction == "short" and trend_direction == "short":
                breakdown["trend_filter"] = True
                score += 1
        
        # c) RSI Agreement (M15): +1
        if direction is not None and pd.notna(rsi_filter_last):
            if direction == "long" and rsi_filter_last > rsi_long_min:
                breakdown["rsi_agrees"] = True
                score += 1
            elif direction == "short" and rsi_filter_last < rsi_short_max:
                breakdown["rsi_agrees"] = True
                score += 1
        
        # d) ATR Validity (M15): +1
        if median_atr_filter > 0 and pd.notna(atr_filter_last):
            atr_low = self.ATR_LOW_MULT * median_atr_filter
            atr_high = self.ATR_HIGH_MULT * median_atr_filter
            if atr_low <= atr_filter_last <= atr_high:
                breakdown["atr_valid"] = True
                score += 1
        
        # e) ADX Validity (H1): +1 (asset-class specific threshold)
        adx_threshold = self._get_adx_threshold(symbol)
        if pd.notna(adx_trend_last) and adx_trend_last >= adx_threshold:
            breakdown["adx_valid"] = True
            score += 1
        
        # ---------------------------------------------------------------------
        # FINAL SIGNAL DETERMINATION
        # ---------------------------------------------------------------------
        result["score"] = score
        result["breakdown"] = breakdown
        
        # Only set direction if score meets asset-class-specific threshold
        min_score = self._get_score_threshold(symbol)
        if direction is not None and score >= min_score:
            result["direction"] = direction
            result["min_score_used"] = min_score  # Track which threshold was used
        else:
            result["direction"] = None
        
        self.last_signals[symbol] = result
        return result
    
    def scan_symbols(self, symbols: list, timeframe: str = "M15") -> list:
        """
        Scan multiple symbols for signals.
        
        Args:
            symbols: List of symbol strings
            timeframe: Chart timeframe (ignored internally)
            
        Returns:
            List of signal dictionaries with direction != None
        """
        signals = []
        
        for symbol in symbols:
            signal = self.generate_signal(symbol, timeframe)
            if signal["direction"] is not None:
                signals.append(signal)
        
        return signals
