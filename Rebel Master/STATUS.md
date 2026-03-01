# REBEL MASTER STATUS
Updated: 2026-02-09 04:35:48

## Core Settings
- Strategy mode: conservative
- Auto trading: False
- SL/TP ATR: 2.0 / 3.0
- Risk per trade (risk engine): 1.0%
- Max open trades: 8
- Max daily drawdown: 0.03

## ML Filter
- Enabled: True
- Model: rf_v1
- Min win prob: 0.55
- Staging enabled: False

## Baseline Locked
- Name: rebel_baseline_locked_v4
- Spread max: < 0.040
- RR min: >= 1.2
- RSI range: 20 < rsi < 80
- EMA bias: ema_fast > ema_slow / ema_fast < ema_slow

## Trade Limiter (Today)
- Trades today total: 40
- Wins / Losses: 266 / 293
- By class: forex:18, crypto:10, metals:7, indices:3, energies:2, softs:0
