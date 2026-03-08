"""
REBEL RULES-BASED AI TRADING SYSTEM
Module: Signal Scoring Engine
Pure rules-based scoring - NO ML
"""

from typing import Dict, Any, Tuple


class SignalScorer:
    """
    Rules-based signal scoring engine.
    Scores trading opportunities from 0-100 based on defined rules.
    """
    
    # Scoring weights (total = 100)
    WEIGHT_TREND = 35
    WEIGHT_MOMENTUM = 25
    WEIGHT_PATTERN = 20
    WEIGHT_STRUCTURE = 20
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score_to_trade = self.config.get('min_score_to_trade', 60)
    
    def score_signal(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a trading signal based on technical indicators.
        
        Args:
            indicators: Dictionary from compute_all_indicators()
        
        Returns:
            Dictionary with scores and trade recommendation
        """
        # Calculate component scores
        trend_score, trend_direction = self._score_trend(indicators)
        momentum_score, momentum_bias = self._score_momentum(indicators)
        pattern_score, pattern_bias = self._score_patterns(indicators)
        structure_score, structure_bias = self._score_structure(indicators)
        
        # Volatility and risk adjustments
        volatility_penalty = self._volatility_penalty(indicators)
        risk_penalty = self._risk_penalty(indicators)
        
        # Calculate raw total
        raw_score = trend_score + momentum_score + pattern_score + structure_score
        
        # Apply penalties
        final_score = max(0, raw_score - volatility_penalty - risk_penalty)
        final_score = min(100, final_score)
        
        # Determine overall bias
        biases = [
            (trend_direction, trend_score),
            (momentum_bias, momentum_score),
            (pattern_bias, pattern_score),
            (structure_bias, structure_score)
        ]
        
        direction = self._determine_direction(biases, indicators)
        confidence = self._calculate_confidence(final_score, direction, indicators)
        
        return {
            "score": int(final_score),
            "direction": direction,
            "confidence": confidence,
            "components": {
                "trend": {"score": trend_score, "max": self.WEIGHT_TREND, "direction": trend_direction},
                "momentum": {"score": momentum_score, "max": self.WEIGHT_MOMENTUM, "bias": momentum_bias},
                "pattern": {"score": pattern_score, "max": self.WEIGHT_PATTERN, "bias": pattern_bias},
                "structure": {"score": structure_score, "max": self.WEIGHT_STRUCTURE, "bias": structure_bias}
            },
            "penalties": {
                "volatility": volatility_penalty,
                "risk": risk_penalty
            },
            "tradeable": final_score >= self.min_score_to_trade and direction != "HOLD"
        }
    
    def _score_trend(self, indicators: Dict[str, Any]) -> Tuple[float, str]:
        """
        Score trend component (0-35 points).
        
        Rules:
        - STRONG_UP/STRONG_DOWN with ADX > 25: Full points
        - UP/DOWN with confirming momentum: Most points
        - SIDEWAYS: Few points
        """
        trend = indicators.get('trend', 'SIDEWAYS')
        trend_strength = indicators.get('trend_strength', 'WEAK')
        adx = indicators.get('adx14', 25)
        
        score = 0
        direction = "NEUTRAL"
        
        if trend == "STRONG_UP":
            score = 30
            direction = "BULLISH"
            if trend_strength == "STRONG":
                score = 35
        
        elif trend == "UP":
            score = 22
            direction = "BULLISH"
            if adx > 25:
                score = 27
        
        elif trend == "STRONG_DOWN":
            score = 30
            direction = "BEARISH"
            if trend_strength == "STRONG":
                score = 35
        
        elif trend == "DOWN":
            score = 22
            direction = "BEARISH"
            if adx > 25:
                score = 27
        
        else:  # SIDEWAYS
            score = 8
            direction = "NEUTRAL"
            # Slight bonus if range-bound but clear
            if trend_strength == "WEAK":
                score = 5
        
        return score, direction
    
    def _score_momentum(self, indicators: Dict[str, Any]) -> Tuple[float, str]:
        """
        Score momentum component (0-25 points).
        
        Rules:
        - RSI in healthy range (40-60) with trend: Good
        - RSI extreme but with MACD confirmation: OK
        - Divergence between RSI and price: Penalty
        """
        rsi = indicators.get('rsi14', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        momentum = indicators.get('momentum', 'NEUTRAL')
        
        score = 0
        bias = "NEUTRAL"
        
        # === BULLISH MOMENTUM ===
        if momentum in ["STRONG_BULLISH", "BULLISH"]:
            bias = "BULLISH"
            
            # Best: RSI 50-65 with positive MACD
            if 50 <= rsi <= 65 and macd_hist > 0:
                score = 25 if momentum == "STRONG_BULLISH" else 22
            
            # Good: RSI rising but not overbought
            elif 40 <= rsi < 70 and macd_hist > 0:
                score = 18
            
            # Caution: Approaching overbought
            elif 70 <= rsi < 80:
                score = 10
            
            # Warning: Overbought
            elif rsi >= 80:
                score = 5
                bias = "NEUTRAL"  # Don't buy overbought
            
            else:
                score = 12
        
        # === BEARISH MOMENTUM ===
        elif momentum in ["STRONG_BEARISH", "BEARISH"]:
            bias = "BEARISH"
            
            # Best: RSI 35-50 with negative MACD
            if 35 <= rsi <= 50 and macd_hist < 0:
                score = 25 if momentum == "STRONG_BEARISH" else 22
            
            # Good: RSI falling but not oversold
            elif 30 < rsi <= 60 and macd_hist < 0:
                score = 18
            
            # Caution: Approaching oversold
            elif 20 < rsi <= 30:
                score = 10
            
            # Warning: Oversold
            elif rsi <= 20:
                score = 5
                bias = "NEUTRAL"  # Don't sell oversold
            
            else:
                score = 12
        
        # === NEUTRAL MOMENTUM ===
        else:
            score = 8
            bias = "NEUTRAL"
        
        return score, bias
    
    def _score_patterns(self, indicators: Dict[str, Any]) -> Tuple[float, str]:
        """
        Score candlestick pattern component (0-20 points).
        
        Rules:
        - Strong reversal patterns (Engulfing, Morning/Evening Star): High score
        - Single candle patterns (Hammer, Shooting Star): Medium score
        - Indecision patterns (Doji): Low score but confirmation needed
        """
        patterns_data = indicators.get('patterns', {})
        patterns = patterns_data.get('patterns', [])
        pattern_signal = patterns_data.get('signal')
        
        if not patterns:
            return 5, "NEUTRAL"  # No pattern = minimal score
        
        score = 0
        bias = pattern_signal or "NEUTRAL"
        
        # Strong reversal patterns
        if "BULLISH_ENGULFING" in patterns:
            score = max(score, 18)
            bias = "BULLISH"
        if "BEARISH_ENGULFING" in patterns:
            score = max(score, 18)
            bias = "BEARISH"
        if "MORNING_STAR" in patterns:
            score = max(score, 20)
            bias = "BULLISH"
        if "EVENING_STAR" in patterns:
            score = max(score, 20)
            bias = "BEARISH"
        
        # Single candle patterns
        if "HAMMER" in patterns:
            score = max(score, 14)
            bias = "BULLISH"
        if "SHOOTING_STAR" in patterns:
            score = max(score, 14)
            bias = "BEARISH"
        
        # Indecision patterns
        if "DOJI" in patterns and score == 0:
            score = 6
            bias = "NEUTRAL"
        
        return score, bias
    
    def _score_structure(self, indicators: Dict[str, Any]) -> Tuple[float, str]:
        """
        Score market structure component (0-20 points).
        
        Rules:
        - Bullish structure + Uptrend: High score
        - Bearish structure + Downtrend: High score
        - Conflicting signals: Low score
        """
        structure_data = indicators.get('market_structure', {})
        structure = structure_data.get('structure', 'UNKNOWN')
        higher_highs = structure_data.get('higher_highs', False)
        lower_lows = structure_data.get('lower_lows', False)
        trend = indicators.get('trend', 'SIDEWAYS')
        
        score = 0
        bias = "NEUTRAL"
        
        # === BULLISH STRUCTURE ===
        if structure == "BULLISH":
            bias = "BULLISH"
            if trend in ["STRONG_UP", "UP"]:
                score = 20  # Structure confirms trend
            else:
                score = 12  # Structure vs trend divergence
        
        # === BEARISH STRUCTURE ===
        elif structure == "BEARISH":
            bias = "BEARISH"
            if trend in ["STRONG_DOWN", "DOWN"]:
                score = 20  # Structure confirms trend
            else:
                score = 12  # Structure vs trend divergence
        
        # === EXPANDING (Volatility breakout) ===
        elif structure == "EXPANDING":
            score = 10
            # Bias based on which way it's breaking
            if higher_highs and not lower_lows:
                bias = "BULLISH"
            elif lower_lows and not higher_highs:
                bias = "BEARISH"
        
        # === RANGING ===
        else:
            score = 6
            bias = "NEUTRAL"
        
        return score, bias
    
    def _volatility_penalty(self, indicators: Dict[str, Any]) -> float:
        """Calculate penalty for extreme volatility."""
        volatility = indicators.get('volatility', 'NORMAL')
        
        if volatility == "EXTREME":
            return 20  # Heavy penalty
        elif volatility == "HIGH":
            return 8
        elif volatility == "LOW":
            return 3  # Slight penalty for no movement
        else:
            return 0
    
    def _risk_penalty(self, indicators: Dict[str, Any]) -> float:
        """Calculate penalty for high risk conditions."""
        risk = indicators.get('risk_level', 'NORMAL')
        
        if risk == "EXTREME":
            return 25  # Severe penalty
        elif risk == "HIGH":
            return 12
        else:
            return 0
    
    def _determine_direction(
        self,
        biases: list,
        indicators: Dict[str, Any]
    ) -> str:
        """
        Determine overall trade direction from component biases.
        
        Rules:
        - Need majority agreement for directional trade
        - Conflicting signals = HOLD
        - High risk = HOLD
        """
        risk = indicators.get('risk_level', 'NORMAL')
        
        # High risk = always HOLD
        if risk in ["EXTREME", "HIGH"]:
            return "HOLD"
        
        # Count weighted biases
        bullish_weight = sum(score for bias, score in biases if bias == "BULLISH")
        bearish_weight = sum(score for bias, score in biases if bias == "BEARISH")
        
        # Need clear majority
        total = bullish_weight + bearish_weight
        
        if total == 0:
            return "HOLD"
        
        bullish_pct = bullish_weight / total
        bearish_pct = bearish_weight / total
        
        # Need 60%+ agreement for directional trade
        if bullish_pct >= 0.6:
            return "BUY"
        elif bearish_pct >= 0.6:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_confidence(
        self,
        score: float,
        direction: str,
        indicators: Dict[str, Any]
    ) -> int:
        """
        Calculate confidence level (0-100).
        
        Rules:
        - Based on score but adjusted for conditions
        - HOLD = 0 confidence
        - Risk adjustments reduce confidence
        """
        if direction == "HOLD":
            return 0
        
        # Base confidence from score
        confidence = score
        
        # Bonus for trend confirmation
        trend = indicators.get('trend', 'SIDEWAYS')
        if direction == "BUY" and trend in ["STRONG_UP", "UP"]:
            confidence += 5
        elif direction == "SELL" and trend in ["STRONG_DOWN", "DOWN"]:
            confidence += 5
        
        # Penalty for weak trend
        trend_strength = indicators.get('trend_strength', 'WEAK')
        if trend_strength == "NO_TREND":
            confidence -= 10
        elif trend_strength == "WEAK":
            confidence -= 5
        
        # Cap at 100
        return max(0, min(100, int(confidence)))


def create_scorer(config: Dict[str, Any] = None) -> SignalScorer:
    """Factory function to create a SignalScorer."""
    return SignalScorer(config)


