# REBEL RULES-BASED TRADING SYSTEM
## Master System Specification (No ML/AI Required)

---

## 📊 OVERVIEW

This document defines a complete **rules-based trading system** extracted from the REBEL Hybrid Engine. 
All decisions are based on mathematical indicators and logical rules - **no machine learning or AI required**.

---

## 🔧 TECHNICAL INDICATORS

### Core Indicators (Computed Every Scan)

| Indicator | Formula | Purpose |
|-----------|---------|---------|
| **EMA9** | 9-period Exponential Moving Average | Fast trend |
| **EMA21** | 21-period Exponential Moving Average | Slow trend |
| **RSI14** | 14-period Relative Strength Index | Momentum/Overbought/Oversold |
| **MACD** | EMA(12) - EMA(26), Signal: EMA(9) | Trend momentum |
| **ADX14** | 14-period Average Directional Index | Trend strength |
| **ATR14** | 14-period Average True Range | Volatility measurement |

### Indicator Calculation Details

```
RSI = 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss over 14 periods

MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line

ADX = 100 * |+DI - -DI| / (+DI + -DI), smoothed over 14 periods

ATR = Average of True Range over 14 periods
True Range = MAX(High-Low, |High-PrevClose|, |Low-PrevClose|)
```

---

## 📈 TREND DETERMINATION

### Trend Classification Rules

```
IF EMA9 > EMA21 AND 5-candle slope > +0.1%:
    TREND = "UP"
    
ELSE IF EMA9 < EMA21 AND 5-candle slope < -0.1%:
    TREND = "DOWN"
    
ELSE:
    TREND = "SIDEWAYS"
```

**Slope Calculation:**
```
slope = (close[-1] - close[-5]) / close[-5]
threshold = 0.001 (0.1%)
```

---

## 📉 MARKET STRUCTURE

### Structure Classification

| Structure | Condition |
|-----------|-----------|
| **BULLISH** | Higher highs, NO lower lows (last 20 vs previous 20 candles) |
| **BEARISH** | Lower lows, NO higher highs |
| **CHOPPY** | Both higher highs AND lower lows |
| **NEUTRAL** | Neither higher highs nor lower lows |

### Detection Logic

```python
swing_high = MAX(highs[-20:])
swing_low = MIN(lows[-20:])
prev_swing_high = MAX(highs[-40:-20])
prev_swing_low = MIN(lows[-40:-20])

higher_highs = swing_high > prev_swing_high
lower_lows = swing_low < prev_swing_low
```

---

## 🌡️ VOLATILITY CLASSIFICATION

### ATR-Based Volatility

```
ATR_RATIO = ATR14 / Current_Close

IF ATR_RATIO < 0.1%:  → VOLATILITY = "LOW"
IF ATR_RATIO < 0.3%:  → VOLATILITY = "NORMAL"
IF ATR_RATIO < 0.6%:  → VOLATILITY = "HIGH"
IF ATR_RATIO >= 0.6%: → VOLATILITY = "EXTREME"
```

---

## ⚠️ RISK LEVEL CALCULATION

### Risk Scoring System (0-100 points)

| Component | Condition | Points |
|-----------|-----------|--------|
| **ATR Risk** | ATR_RATIO > 0.6% | +40 |
| | ATR_RATIO > 0.3% | +20 |
| | ATR_RATIO > 0.1% | +10 |
| **ADX Risk** | ADX < 20 (choppy) | +30 |
| | ADX < 25 | +15 |
| **Spread Risk** | Spread/Price > 0.1% | +30 |
| | Spread/Price > 0.05% | +15 |

### Risk Classification

```
IF risk_score >= 70: → RISK = "EXTREME"
IF risk_score >= 50: → RISK = "HIGH"
IF risk_score >= 25: → RISK = "NORMAL"
IF risk_score < 25:  → RISK = "LOW"
```

---

## 🕯️ CANDLESTICK PATTERNS

### Pattern Detection Rules

#### Hammer (Bullish Reversal)
```
body = |close - open|
lower_shadow = MIN(open, close) - low
upper_shadow = high - MAX(open, close)

HAMMER = (lower_shadow > 2 * body) AND (upper_shadow < 0.3 * body) AND (body > 0)
```

#### Shooting Star (Bearish Reversal)
```
SHOOTING_STAR = (upper_shadow > 2 * body) AND (lower_shadow < 0.3 * body) AND (body > 0)
```

#### Bullish Engulfing
```
prev_bearish = prev.close < prev.open
curr_bullish = curr.close > curr.open
engulfs = (curr.open < prev.close) AND (curr.close > prev.open)

BULLISH_ENGULFING = prev_bearish AND curr_bullish AND engulfs
```

#### Bearish Engulfing
```
prev_bullish = prev.close > prev.open
curr_bearish = curr.close < curr.open
engulfs = (curr.open > prev.close) AND (curr.close < prev.open)

BEARISH_ENGULFING = prev_bullish AND curr_bearish AND engulfs
```

#### Doji (Indecision)
```
DOJI = body < (0.1 * total_range)
```

---

## 🎯 SIGNAL SCORING SYSTEM (0-100 Points)

### Score Components

| Component | Max Points | Breakdown |
|-----------|------------|-----------|
| **Trend Score** | 40 | Trend alignment + confirmation |
| **Pattern Score** | 20 | Candlestick patterns |
| **Volatility Score** | 20 | Volatility suitability |
| **Structure Score** | 20 | Market structure quality |

### Trend Scoring (0-40 points)

```
IF TREND == "UP":
    score = 20
    IF 50 < RSI < 70: score += 10  # Healthy momentum
    IF MACD_histogram > 0: score += 10  # Confirming
    
IF TREND == "DOWN":
    score = 20
    IF 30 < RSI < 50: score += 10
    IF MACD_histogram < 0: score += 10
    
IF TREND == "SIDEWAYS":
    score = 5
```

### Pattern Scoring (0-20 points)

```
BULLISH_ENGULFING or BEARISH_ENGULFING: +15 points
HAMMER or SHOOTING_STAR: +10 points
DOJI: +5 points
(Only count strongest pattern)
```

### Volatility Scoring (0-20 points)

```
VOLATILITY == "NORMAL": +15 points
VOLATILITY == "LOW": +10 points
VOLATILITY == "HIGH": +5 points
VOLATILITY == "EXTREME": +0 points

RISK == "LOW": +5 bonus
RISK == "NORMAL": +3 bonus
RISK == "HIGH": -5 penalty
RISK == "EXTREME": -10 penalty
```

### Structure Scoring (0-20 points)

```
IF structure == "BULLISH" AND trend == "UP": +15
IF structure == "BEARISH" AND trend == "DOWN": +15
IF structure == "NEUTRAL": +8
IF structure == "CHOPPY": +3

BONUS: Clear higher highs (no lower lows): +5
BONUS: Clear lower lows (no higher highs): +5
```

---

## 🚦 TRADE DECISION RULES

### Entry Conditions

#### BUY Signal Requirements
```
1. SCORE >= 60
2. TREND == "UP" OR pattern in [BULLISH_ENGULFING, HAMMER]
3. RISK NOT IN ["HIGH", "EXTREME"]
4. RSI > 50 AND RSI < 70 (not overbought)
5. ADX > 20 (trend has strength)
6. Market structure == "BULLISH" or "NEUTRAL"
```

#### SELL Signal Requirements
```
1. SCORE >= 60
2. TREND == "DOWN" OR pattern in [BEARISH_ENGULFING, SHOOTING_STAR]
3. RISK NOT IN ["HIGH", "EXTREME"]
4. RSI < 50 AND RSI > 30 (not oversold)
5. ADX > 20 (trend has strength)
6. Market structure == "BEARISH" or "NEUTRAL"
```

#### HOLD (No Trade)
```
- SCORE < 60
- RISK == "HIGH" or "EXTREME"
- RSI > 80 (overbought) or RSI < 20 (oversold)
- ADX < 20 (no clear trend)
- VOLATILITY == "EXTREME"
```

---

## 🛡️ STOP LOSS & TAKE PROFIT

### ATR-Based Levels

```
FOR BUY TRADES:
    Stop Loss = Entry Price - (1.0 × ATR14)
    Take Profit = Entry Price + (2.0 × ATR14)
    
FOR SELL TRADES:
    Stop Loss = Entry Price + (1.0 × ATR14)
    Take Profit = Entry Price - (2.0 × ATR14)
```

### Risk:Reward Ratio
- **Minimum R:R = 1:2** (SL = 1x ATR, TP = 2x ATR)

---

## 💰 POSITION SIZING

### Risk-Based Lot Calculation

```python
# Default: Risk 1% of account per trade
risk_percent = 0.01
risk_amount = account_balance × risk_percent × risk_scale

# Calculate lot size
sl_distance = |entry_price - stop_loss|
risk_per_lot = (sl_distance / tick_size) × tick_value
lot_size = risk_amount / risk_per_lot

# Respect symbol limits
lot_size = MAX(min_lot, MIN(max_lot, lot_size))
lot_size = ROUND(lot_size / lot_step) × lot_step
```

### Risk Scale Multipliers

| Confidence | Risk Scale |
|------------|------------|
| High (score 80+) | 1.5 - 2.0 |
| Medium (score 60-79) | 1.0 |
| Low (score 50-59) | 0.5 - 0.8 |

---

## 📋 SYMBOL GROUPS & RISK SETTINGS

### Default Group Configuration

| Group | Symbols | Risk Scale | Max Trades |
|-------|---------|------------|------------|
| **Metals** | XAUUSD | 1.0 | 1 |
| **Majors** | EURUSD, GBPUSD, USDJPY, AUDUSD | 1.0 | 2 |
| **Crosses** | GBPJPY, CADJPY, EURAUD | 0.8 | 1 |
| **Crypto** | BTCUSD, ETHUSD | 0.5 | 1 |
| **Indices** | US500, NAS100 | 0.7 | 1 |
| **Softs** | COFFEE, SOYBEAN, COCOA | 0.4 | 1 |
| **Energies** | XTIUSD, XBRUSD, XNGUSD | 0.6 | 1 |

---

## ⏰ SESSION ACTIVATION

### Trading Sessions (UTC)

| Session | Hours (UTC) | Groups Active |
|---------|-------------|---------------|
| **Tokyo** | 00:00 - 09:00 | Metals, Majors, Crosses, Crypto |
| **London** | 07:00 - 16:00 | All Groups |
| **New York** | 12:00 - 21:00 | All Groups |
| **Weekend** | Sat-Sun | Crypto only |

### Session Detection

```python
hour = current_utc_hour
weekday = current_weekday  # 0=Mon, 6=Sun

IF weekday >= 5:
    session = "weekend"
ELSE IF 0 <= hour < 9:
    session = "tokyo"
ELSE IF 7 <= hour < 16:
    session = "london"
ELSE IF 12 <= hour < 21:
    session = "newyork"
ELSE:
    session = "closed"
```

---

## 🚨 GLOBAL RISK LIMITS

### Account Protection Rules

| Rule | Threshold | Action |
|------|-----------|--------|
| **Max Total Open Trades** | 6 | Block new trades |
| **Max Drawdown** | 6% | Emergency close ALL positions |
| **Max Risk Scale Per Loop** | 2.0 | Cap combined risk |
| **Emergency Close** | Enabled | Auto-close at max drawdown |

### Drawdown Calculation

```python
drawdown_percent = ((balance - equity) / balance) × 100

IF drawdown > max_drawdown_percent:
    CLOSE_ALL_POSITIONS()
    SKIP_TRADING_THIS_LOOP()
```

---

## 🔄 MAIN LOOP LOGIC (Pseudocode)

```
EVERY 60 SECONDS:
    
    1. CHECK EMERGENCY CONDITIONS
       IF drawdown > 6%:
           close_all_positions()
           continue
    
    2. GET OPEN POSITIONS
       count trades per group
       
    3. FOR EACH SYMBOL:
       
       a. CHECK SESSION ACTIVATION
          IF symbol.group not active in current session:
              skip
       
       b. LOAD 250 CANDLES (M15)
       
       c. COMPUTE INDICATORS
          - EMA9, EMA21, RSI14, MACD, ADX14, ATR14
          - Trend, Structure, Volatility, Risk
          - Patterns
       
       d. CALCULATE SIGNAL SCORE (0-100)
          trend_score + pattern_score + volatility_score + structure_score
       
       e. DETERMINE DIRECTION
          IF score >= 60 AND risk NOT HIGH/EXTREME:
              IF trend == UP AND confirmations:
                  direction = BUY
              ELSE IF trend == DOWN AND confirmations:
                  direction = SELL
          ELSE:
              direction = HOLD
       
       f. CHECK GROUP LIMITS
          IF group_open_trades >= group_max_trades:
              direction = HOLD
       
       g. CHECK GLOBAL LIMITS
          IF total_trades >= 6:
              direction = HOLD
       
       h. EXECUTE TRADE
          IF direction != HOLD:
              calculate_lot_size()
              set_sl_tp()
              place_order()
    
    4. UPDATE DASHBOARD
    
    5. WAIT 60 SECONDS
```

---

## 📊 DECISION MATRIX SUMMARY

### Quick Reference: When to Trade

| Condition | Score | Trend | RSI | ADX | Risk | Action |
|-----------|-------|-------|-----|-----|------|--------|
| Strong Buy | 70+ | UP | 50-65 | >25 | LOW | BUY 1.5x |
| Standard Buy | 60-69 | UP | 50-70 | >20 | NORMAL | BUY 1.0x |
| Strong Sell | 70+ | DOWN | 35-50 | >25 | LOW | SELL 1.5x |
| Standard Sell | 60-69 | DOWN | 30-50 | >20 | NORMAL | SELL 1.0x |
| Weak Signal | 50-59 | ANY | ANY | ANY | NORMAL | HOLD |
| High Risk | ANY | ANY | ANY | ANY | HIGH+ | HOLD |
| Overbought | ANY | UP | >80 | ANY | ANY | HOLD |
| Oversold | ANY | DOWN | <20 | ANY | ANY | HOLD |
| No Trend | <50 | SIDEWAYS | ANY | <20 | ANY | HOLD |

---

## 🎛️ CONFIGURABLE PARAMETERS

### Tunable Settings

```yaml
# Scoring Thresholds
min_score_to_trade: 60
score_for_high_confidence: 70

# Indicator Periods
ema_fast: 9
ema_slow: 21
rsi_period: 14
macd_fast: 12
macd_slow: 26
macd_signal: 9
atr_period: 14
adx_period: 14

# Risk Management
risk_per_trade_percent: 1.0
sl_atr_multiplier: 1.0
tp_atr_multiplier: 2.0
max_drawdown_percent: 6.0
max_total_open_trades: 6

# Session Settings
scan_interval_seconds: 60
candle_history: 250
timeframe: M15
```

---

## ✅ CHECKLIST: Trade Entry

Before placing any trade, verify:

- [ ] Score >= 60
- [ ] Risk level is LOW or NORMAL
- [ ] RSI between 20-80
- [ ] ADX > 20
- [ ] Volatility is not EXTREME
- [ ] Trend direction matches trade direction
- [ ] Group limit not exceeded
- [ ] Global trade limit not exceeded
- [ ] Account drawdown < 6%
- [ ] Session is active for this symbol group

---

## 🏁 IMPLEMENTATION NOTES

1. **No AI/ML Required** - All decisions are mathematical
2. **60-Second Scan Cycle** - Processes all symbols every minute
3. **M15 Timeframe** - Primary timeframe for signals
4. **250 Candles History** - Sufficient for all indicator calculations
5. **Risk-First Approach** - Multiple safety checks before any trade
6. **Fail-Safe Design** - Default to HOLD when uncertain

---

*Last Updated: December 27, 2025*
*Extracted from REBEL Hybrid Trading Engine*

