"""
REBEL RULES-BASED AI TRADING SYSTEM
Module: Scanner Bridge
Bridges the Intelligent Scanner with the Rules-Based Engine
"""

import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Add Python folder to path for scanner imports
PYTHON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)


class ScannerBridge:
    """
    Bridge between the Intelligent Scanner and the Rules-Based Engine.
    
    Allows the engine to use scanner signals while keeping
    the scanner usable as a standalone tool.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scanner_available = False
        self.helpers_available = False
        self.aim_available = False
        
        # Try to import scanner components
        self._init_scanner()
    
    def _init_scanner(self) -> None:
        """Initialize scanner components."""
        try:
            # Import helpers
            from rebel_helpers import (
                calculate_indicators,
                detect_trend,
                classify_volatility,
                detect_patterns,
                calculate_risk,
                generate_sparkline,
                ensure_json_serializable,
                map_symbols,
                get_spread_ratio,
                check_safety
            )
            self.helpers = {
                'calculate_indicators': calculate_indicators,
                'detect_trend': detect_trend,
                'classify_volatility': classify_volatility,
                'detect_patterns': detect_patterns,
                'calculate_risk': calculate_risk,
                'generate_sparkline': generate_sparkline,
                'ensure_json_serializable': ensure_json_serializable,
                'map_symbols': map_symbols,
                'get_spread_ratio': get_spread_ratio,
                'check_safety': check_safety
            }
            self.helpers_available = True
            print("  [OK] Scanner helpers loaded")
            
        except ImportError as e:
            print(f"  [!] Scanner helpers not available: {e}")
            self.helpers = {}
        
        try:
            # Import AIM
            from Rebel_AIM import get_ai_signal
            self.get_ai_signal = get_ai_signal
            self.aim_available = True
            print("  [OK] Rebel AIM loaded")
            
        except ImportError as e:
            print(f"  [!] Rebel AIM not available: {e}")
            self.get_ai_signal = None
        
        self.scanner_available = self.helpers_available
    
    def scan_symbol(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
        use_aim: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Scan a single symbol using scanner logic.
        
        Args:
            symbol: Trading symbol
            candles: List of candle dictionaries (OHLCV)
            use_aim: Whether to use AIM for AI signal
        
        Returns:
            Scanner result dictionary or None if unavailable
        """
        if not self.helpers_available:
            return None
        
        if len(candles) < 30:
            return None
        
        try:
            # Calculate indicators using scanner's method
            indicators = self.helpers['calculate_indicators'](candles)
            
            if not indicators:
                return None
            
            # Extract values
            trend = indicators.get("trend", "UNKNOWN")
            volatility = indicators.get("volatility_class", "NORMAL")
            pattern = indicators.get("pattern", "none")
            adx = indicators.get("adx", 0.0)
            atr_ratio = indicators.get("volatility_ratio", 1.0)
            rsi = indicators.get("rsi", 50.0)
            current_price = indicators.get("current_price", 0.0)
            
            # Generate sparkline
            sparkline = self.helpers['generate_sparkline'](candles, length=16)
            
            # Check safety (need broker symbol)
            broker_symbol = symbol  # Assume same for now
            spread, spread_ratio = self.helpers['get_spread_ratio'](broker_symbol)
            safety, safety_reason = self.helpers['check_safety'](
                broker_symbol, volatility, spread_ratio
            )
            
            # Calculate risk
            risk = self.helpers['calculate_risk'](
                volatility, spread_ratio, atr_ratio, adx, pattern
            )
            
            # Build result
            result = {
                "symbol": symbol,
                "trend": trend,
                "volatility": volatility,
                "pattern": pattern,
                "adx": adx,
                "rsi": rsi,
                "atr_ratio": atr_ratio,
                "current_price": current_price,
                "sparkline": sparkline,
                "spread_ratio": spread_ratio,
                "safety": safety,
                "safety_reason": safety_reason,
                "risk": risk,
                "indicators": indicators
            }
            
            # Get AIM signal if available and requested
            if use_aim and self.aim_available and self.get_ai_signal and safety == "SAFE":
                snapshot = self._build_snapshot(symbol, candles, indicators)
                ai_result = self.get_ai_signal(symbol, snapshot)
                
                result["direction"] = ai_result.get("direction", "HOLD")
                result["confidence"] = ai_result.get("confidence", 0)
                result["ai_comment"] = ai_result.get("comment", "")
            else:
                # No AIM - derive direction from indicators
                result["direction"] = self._derive_direction(indicators)
                result["confidence"] = self._derive_confidence(indicators, result["direction"])
                result["ai_comment"] = ""
            
            return result
            
        except Exception as e:
            print(f"  Scanner bridge error for {symbol}: {e}")
            return None
    
    def scan_symbols(
        self,
        symbols: List[str],
        candles_map: Dict[str, List[Dict]],
        use_aim: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scan multiple symbols.
        
        Args:
            symbols: List of trading symbols
            candles_map: Dictionary of symbol -> candles list
            use_aim: Whether to use AIM
        
        Returns:
            List of scanner results
        """
        results = []
        
        for symbol in symbols:
            candles = candles_map.get(symbol, [])
            if candles:
                result = self.scan_symbol(symbol, candles, use_aim)
                if result:
                    results.append(result)
        
        # Sort by confidence
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return results
    
    def _build_snapshot(
        self,
        symbol: str,
        candles: List[Dict],
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build snapshot for AIM."""
        last_candle = candles[-1] if candles else {}
        
        snapshot = {
            "symbol": str(symbol),
            "broker_symbol": str(symbol),
            "time": datetime.now(timezone.utc).isoformat(),
            "last_candle": last_candle,
            "candle_count": len(candles),
            "indicators": indicators,
            "trend": str(indicators.get("trend", "UNKNOWN")),
            "volatility": str(indicators.get("volatility_class", "NORMAL")),
            "volatility_ratio": float(indicators.get("volatility_ratio", 1.0)),
            "pattern": str(indicators.get("pattern", "none")),
            "structure": indicators.get("structure", {}),
            "rsi": float(indicators.get("rsi", 50.0)),
            "macd": float(indicators.get("macd", 0.0)),
            "adx": float(indicators.get("adx", 0.0)),
            "atr": float(indicators.get("atr", 0.0)),
            "current_price": float(indicators.get("current_price", 0.0))
        }
        
        if self.helpers_available:
            snapshot = self.helpers['ensure_json_serializable'](snapshot)
        
        return snapshot
    
    def _derive_direction(self, indicators: Dict[str, Any]) -> str:
        """Derive direction from indicators when AIM is not used."""
        trend = indicators.get("trend", "UNKNOWN")
        rsi = indicators.get("rsi", 50.0)
        pattern = indicators.get("pattern", "none").lower()
        adx = indicators.get("adx", 0.0)
        
        # Need trend confirmation
        if adx < 20:
            return "HOLD"  # No clear trend
        
        # Check RSI extremes
        if rsi > 80 or rsi < 20:
            return "HOLD"  # Overbought/oversold
        
        # Bullish conditions
        if trend == "UP" and rsi < 70:
            if "hammer" in pattern or "engulfing" in pattern:
                return "BUY"
            if rsi > 50:
                return "BUY"
        
        # Bearish conditions
        if trend == "DOWN" and rsi > 30:
            if "shooting" in pattern or "engulfing" in pattern:
                return "SELL"
            if rsi < 50:
                return "SELL"
        
        return "HOLD"
    
    def _derive_confidence(self, indicators: Dict[str, Any], direction: str) -> int:
        """Derive confidence score from indicators."""
        if direction == "HOLD":
            return 0
        
        score = 50  # Base
        
        trend = indicators.get("trend", "UNKNOWN")
        adx = indicators.get("adx", 0.0)
        rsi = indicators.get("rsi", 50.0)
        pattern = indicators.get("pattern", "none").lower()
        
        # ADX strength
        if adx > 40:
            score += 15
        elif adx > 25:
            score += 10
        
        # RSI in healthy range
        if direction == "BUY" and 50 <= rsi <= 65:
            score += 10
        elif direction == "SELL" and 35 <= rsi <= 50:
            score += 10
        
        # Pattern confirmation
        if "engulfing" in pattern:
            score += 15
        elif "hammer" in pattern or "shooting" in pattern:
            score += 10
        
        # Trend alignment
        if (direction == "BUY" and trend == "UP") or (direction == "SELL" and trend == "DOWN"):
            score += 10
        
        return min(100, max(0, score))
    
    def get_top_signals(
        self,
        results: List[Dict[str, Any]],
        min_confidence: int = 50,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top signals from scan results."""
        filtered = [
            r for r in results
            if r.get('direction') != 'HOLD'
            and r.get('confidence', 0) >= min_confidence
            and r.get('safety') == 'SAFE'
        ]
        
        return filtered[:limit]


def create_scanner_bridge(config: Dict[str, Any] = None) -> ScannerBridge:
    """Factory function to create a ScannerBridge."""
    return ScannerBridge(config)


