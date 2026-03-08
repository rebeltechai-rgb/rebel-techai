"""
REBEL RULES-BASED AI TRADING SYSTEM
Module: AI Decision Brain
Uses GPT to interpret and apply trading rules - NOT machine learning
"""

import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI


# =============================================================================
# SYSTEM RULES (Embedded in AI prompts)
# =============================================================================

TRADING_RULES = """
## REBEL TRADING RULES - YOU MUST FOLLOW THESE EXACTLY

### ENTRY RULES

**BUY CONDITIONS (ALL must be met):**
1. Signal score >= 75
2. Trend is UP or STRONG_UP
3. RSI between 35-70 (not overbought)
4. ADX >= MIN_ADX (trend has strength)
5. Risk level is LOW or NORMAL (never HIGH or EXTREME)
6. At least 2 of 3 confirmations: Bullish pattern, Bullish structure, Positive MACD

**SELL CONDITIONS (ALL must be met):**
1. Signal score >= 75
2. Trend is DOWN or STRONG_DOWN
3. RSI between 30-65 (not oversold)
4. ADX >= MIN_ADX (trend has strength)
5. Risk level is LOW or NORMAL (never HIGH or EXTREME)
6. At least 2 of 3 confirmations: Bearish pattern, Bearish structure, Negative MACD

### EXIT RULES

**Stop Loss:**
- Standard: 1.0 × ATR from entry
- Tight (high confidence): 0.8 × ATR from entry
- Wide (volatile market): 1.5 × ATR from entry

**Take Profit:**
- Minimum R:R ratio: 1:2 (TP = 2 × SL distance)
- Standard: 2.0 × ATR from entry
- Extended (strong trend): 3.0 × ATR from entry

### HOLD CONDITIONS (DO NOT TRADE)
- Score < 75
- Risk level is HIGH or EXTREME
- RSI > 80 (overbought) or RSI < 20 (oversold)
- ADX < MIN_ADX (no clear trend)
- Volatility is EXTREME
- Conflicting signals (trend vs momentum vs structure disagree)

### RISK SCALING
- Score 90+, Low Risk: risk_scale = 1.5
- Score 80-89, Normal Risk: risk_scale = 1.2
- Score 75-79, Normal Risk: risk_scale = 1.0
- Score 75-79, any concerns: risk_scale = 0.8

### NEVER DO:
- Trade against the trend
- Enter when RSI is extreme (>80 or <20)
- Trade during EXTREME volatility
- Ignore the risk level classification
- Make decisions based on "gut feeling" - only use the data provided
"""


class AIBrain:
    """
    AI Decision Brain using GPT with strict rule enforcement.
    
    This is NOT machine learning - it's using AI to interpret
    and apply pre-defined trading rules.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = self.config.get('model', 'gpt-4o')
        self.temperature = self.config.get('temperature', 0.1)  # Low temp for consistency
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def make_decision(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        score_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make a trading decision based on indicators and score.
        
        Args:
            symbol: Trading symbol
            indicators: From compute_all_indicators()
            score_data: From SignalScorer.score_signal()
        
        Returns:
            Decision dictionary with direction, confidence, SL, TP, etc.
        """
        try:
            # Build the analysis prompt
            prompt = self._build_decision_prompt(symbol, indicators, score_data)
            
            # Call GPT
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            decision = json.loads(content)
            
            # Validate and enforce rules
            decision = self._validate_decision(decision, indicators, score_data)
            
            return decision
            
        except Exception as e:
            # Safe fallback - HOLD on any error
            return self._safe_hold(f"AI Error: {str(e)[:100]}")
    
    def analyze_market(
        self,
        symbol: str,
        indicators: Dict[str, Any]
    ) -> str:
        """
        Get AI interpretation of market conditions (no trade decision).
        
        Args:
            symbol: Trading symbol
            indicators: From compute_all_indicators()
        
        Returns:
            Natural language market analysis
        """
        try:
            prompt = self._build_analysis_prompt(symbol, indicators)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model for analysis
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical analyst. Provide brief, factual analysis of the market conditions. Do not make trading recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Analysis unavailable: {str(e)[:50]}"
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt with embedded rules."""
        return f"""You are the REBEL Trading System AI Brain.

Your job is to analyze technical data and make trading decisions based on STRICT RULES.
You are NOT doing machine learning or prediction - you are applying predefined rules.

{TRADING_RULES}

RESPONSE FORMAT:
Return ONLY valid JSON in this exact format:
{{
    "direction": "BUY" | "SELL" | "HOLD",
    "confidence": 0-100,
    "sl_atr_multiplier": 0.8 | 1.0 | 1.5,
    "tp_atr_multiplier": 2.0 | 2.5 | 3.0,
    "risk_scale": 0.5-1.5,
    "reasoning": "Brief explanation of which rules were applied",
    "warnings": ["list of any concerns"]
}}

BE CONSERVATIVE. When in doubt, HOLD."""
    
    def _build_decision_prompt(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        score_data: Dict[str, Any]
    ) -> str:
        """Build the decision request prompt."""
        
        # Extract key data
        trend = indicators.get('trend', 'UNKNOWN')
        trend_strength = indicators.get('trend_strength', 'UNKNOWN')
        rsi = indicators.get('rsi14', 50)
        adx = indicators.get('adx14', 25)
        adx_min = score_data.get('adx_min', 20)
        macd_hist = indicators.get('macd_histogram', 0)
        volatility = indicators.get('volatility', 'NORMAL')
        risk_level = indicators.get('risk_level', 'NORMAL')
        
        structure = indicators.get('market_structure', {})
        structure_type = structure.get('structure', 'UNKNOWN')
        
        patterns = indicators.get('patterns', {})
        pattern_list = patterns.get('patterns', [])
        pattern_signal = patterns.get('signal')
        
        momentum = indicators.get('momentum', 'NEUTRAL')
        
        atr = indicators.get('atr14', 0)
        close = indicators.get('close', 0)
        
        # Score components
        score = score_data.get('score', 0)
        score_direction = score_data.get('direction', 'HOLD')
        score_confidence = score_data.get('confidence', 0)
        components = score_data.get('components', {})
        
        prompt = f"""
TRADING DECISION REQUEST FOR: {symbol}

=== CURRENT PRICE DATA ===
Close: {close}
ATR(14): {atr:.5f}

=== SIGNAL SCORE ===
Total Score: {score}/100
Pre-computed Direction: {score_direction}
Pre-computed Confidence: {score_confidence}%

Score Components:
- Trend: {components.get('trend', {}).get('score', 0)}/{components.get('trend', {}).get('max', 35)}
- Momentum: {components.get('momentum', {}).get('score', 0)}/{components.get('momentum', {}).get('max', 25)}
- Pattern: {components.get('pattern', {}).get('score', 0)}/{components.get('pattern', {}).get('max', 20)}
- Structure: {components.get('structure', {}).get('score', 0)}/{components.get('structure', {}).get('max', 20)}

=== TECHNICAL INDICATORS ===
Trend: {trend} (Strength: {trend_strength})
RSI(14): {rsi:.1f}
ADX(14): {adx:.1f}
Min ADX Required: {adx_min}
MACD Histogram: {macd_hist:.6f}
Momentum: {momentum}
Volatility: {volatility}
Risk Level: {risk_level}

=== MARKET STRUCTURE ===
Structure: {structure_type}
Higher Highs: {structure.get('higher_highs', False)}
Lower Lows: {structure.get('lower_lows', False)}

=== CANDLESTICK PATTERNS ===
Patterns: {', '.join(pattern_list) if pattern_list else 'None'}
Pattern Signal: {pattern_signal or 'None'}

=== DECISION REQUEST ===
Apply the trading rules to this data and return your decision as JSON.
Remember: Score must be >= 75 to trade. Risk must be LOW or NORMAL.
Min ADX Required: {adx_min}
"""
        return prompt
    
    def _build_analysis_prompt(self, symbol: str, indicators: Dict[str, Any]) -> str:
        """Build a market analysis prompt (no decision)."""
        
        trend = indicators.get('trend', 'UNKNOWN')
        rsi = indicators.get('rsi14', 50)
        adx = indicators.get('adx14', 25)
        volatility = indicators.get('volatility', 'NORMAL')
        structure = indicators.get('market_structure', {}).get('structure', 'UNKNOWN')
        patterns = indicators.get('patterns', {}).get('patterns', [])
        
        return f"""
Analyze the current market conditions for {symbol}:

- Trend: {trend}
- RSI: {rsi:.1f}
- ADX: {adx:.1f}
- Volatility: {volatility}
- Structure: {structure}
- Patterns: {', '.join(patterns) if patterns else 'None'}

Provide a 2-3 sentence factual summary of what the indicators show.
"""
    
    def _validate_decision(
        self,
        decision: Dict[str, Any],
        indicators: Dict[str, Any],
        score_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and enforce rules on AI decision.
        
        This is the safety layer - even if AI makes a mistake,
        we enforce the rules here.
        """
        direction = decision.get('direction', 'HOLD').upper()
        risk_level = indicators.get('risk_level', 'NORMAL')
        score = score_data.get('score', 0)
        rsi = indicators.get('rsi14', 50)
        adx = indicators.get('adx14', 25)
        adx_min = score_data.get('adx_min', 20)
        volatility = indicators.get('volatility', 'NORMAL')
        
        warnings = decision.get('warnings', [])
        
        # === HARD RULES - Override AI if violated ===
        
        # Rule 1: Score must be >= 75 to trade
        if score < 75 and direction != "HOLD":
            direction = "HOLD"
            warnings.append(f"Overridden: Score {score} < 75")
        
        # Rule 2: Risk must be LOW or NORMAL
        if risk_level in ["HIGH", "EXTREME"] and direction != "HOLD":
            direction = "HOLD"
            warnings.append(f"Overridden: Risk level {risk_level}")
        
        # Rule 3: RSI extremes
        if direction == "BUY" and rsi > 80:
            direction = "HOLD"
            warnings.append(f"Overridden: RSI {rsi:.0f} overbought")
        if direction == "SELL" and rsi < 20:
            direction = "HOLD"
            warnings.append(f"Overridden: RSI {rsi:.0f} oversold")
        
        # Rule 4: ADX must show trend
        if adx < adx_min and direction != "HOLD":
            direction = "HOLD"
            warnings.append(f"Overridden: ADX {adx:.0f} < {adx_min}")
        
        # Rule 5: No trading in extreme volatility
        if volatility == "EXTREME" and direction != "HOLD":
            direction = "HOLD"
            warnings.append("Overridden: Extreme volatility")
        
        # === Validate numeric fields ===
        confidence = decision.get('confidence', 0)
        confidence = max(0, min(100, int(confidence)))
        
        if direction == "HOLD":
            confidence = 0
        
        sl_mult = decision.get('sl_atr_multiplier', 1.0)
        tp_mult = decision.get('tp_atr_multiplier', 2.0)
        risk_scale = decision.get('risk_scale', 1.0)
        
        # Clamp values
        sl_mult = max(0.5, min(2.0, float(sl_mult)))
        tp_mult = max(1.5, min(4.0, float(tp_mult)))
        risk_scale = max(0.5, min(1.5, float(risk_scale)))
        
        # Calculate actual SL/TP prices
        close = indicators.get('close', 0)
        atr = indicators.get('atr14', 0)
        
        if direction == "BUY" and atr > 0:
            sl = close - (sl_mult * atr)
            tp = close + (tp_mult * atr)
        elif direction == "SELL" and atr > 0:
            sl = close + (sl_mult * atr)
            tp = close - (tp_mult * atr)
        else:
            sl = None
            tp = None
        
        return {
            "direction": direction,
            "confidence": confidence,
            "sl": sl,
            "tp": tp,
            "sl_atr_multiplier": sl_mult,
            "tp_atr_multiplier": tp_mult,
            "risk_scale": risk_scale,
            "reasoning": decision.get('reasoning', ''),
            "warnings": warnings
        }
    
    def _safe_hold(self, reason: str) -> Dict[str, Any]:
        """Return a safe HOLD decision."""
        return {
            "direction": "HOLD",
            "confidence": 0,
            "sl": None,
            "tp": None,
            "sl_atr_multiplier": 1.0,
            "tp_atr_multiplier": 2.0,
            "risk_scale": 1.0,
            "reasoning": reason,
            "warnings": [reason]
        }


def create_ai_brain(config: Dict[str, Any] = None) -> AIBrain:
    """Factory function to create an AIBrain."""
    return AIBrain(config)


