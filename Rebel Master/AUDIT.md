# REBEL MASTER AUDIT
Updated: 2026-02-09 06:06:43

## Files modified in last 24h

- C:\Rebel Technologies\Rebel Master\Config\service_state.json | 02/09/2026 06:06:42
- C:\Rebel Technologies\Rebel Master\logs\regime_guard_state.json | 02/09/2026 06:06:41
- C:\Rebel Technologies\Rebel Master\logs\filter_skip_state.json | 02/09/2026 06:06:41
- C:\Rebel Technologies\Rebel Master\logs\filter_rejections.txt | 02/09/2026 06:06:41
- C:\Rebel Technologies\Rebel Master\Python\service_state.json | 02/09/2026 06:06:41
- C:\Rebel Technologies\Rebel Master\Python\rebel_profit_lock.py | 02/09/2026 05:58:10
- C:\Rebel Technologies\Rebel Master\logs\trade_limiter_state.json | 02/09/2026 05:51:34
- C:\Rebel Technologies\Rebel Master\logs\trade_limiter.txt | 02/09/2026 05:51:34
- C:\Rebel Technologies\Rebel Master\ML\trade_features.csv | 02/09/2026 05:51:34
- C:\Rebel Technologies\Rebel Master\logs\trades.csv | 02/09/2026 05:51:34
- C:\Rebel Technologies\Rebel Master\ML\features.csv | 02/09/2026 05:51:34
- C:\Rebel Technologies\Rebel Master\ML\training_dataset.csv | 02/09/2026 05:37:11
- C:\Rebel Technologies\Rebel Master\Python\performance_state.json | 02/09/2026 05:32:09
- C:\Rebel Technologies\Rebel Master\Config\master_config.yaml | 02/09/2026 05:26:30
- C:\Rebel Technologies\Rebel Master\Config\ml_milestone_state.json | 02/09/2026 05:22:25
- C:\Rebel Technologies\Rebel Master\Python\__pycache__\rebel_profit_lock.cpython-312.pyc | 02/09/2026 05:17:11
- C:\Rebel Technologies\Rebel Master\ML\rebel_baseline_locked.py | 02/09/2026 05:11:02
- C:\Rebel Technologies\Rebel Master\ML\labels.csv | 02/09/2026 04:56:50
- C:\Rebel Technologies\Rebel Master\Config\tracked_positions.json | 02/09/2026 04:56:50
- C:\Rebel Technologies\Rebel Master\Python\__pycache__\rebel_signals.cpython-312.pyc | 02/09/2026 04:56:48
- C:\Rebel Technologies\Rebel Master\Python\__pycache__\rebel_signal_filters.cpython-312.pyc | 02/09/2026 04:56:48
- C:\Rebel Technologies\Rebel Master\Python\rebel_signals.py | 02/09/2026 04:56:13
- C:\Rebel Technologies\Rebel Master\Python\rebel_signal_filters.py | 02/09/2026 04:42:16
- C:\Rebel Technologies\Rebel Master\ML\system_status_report.py | 02/09/2026 04:36:02
- C:\Rebel Technologies\Rebel Master\logs\symbol_stats_report.csv | 02/09/2026 03:17:16
- C:\Rebel Technologies\Rebel Master\ML\symbol_stats_report.py | 02/09/2026 03:16:41
- C:\Rebel Technologies\Rebel Master\Python\__pycache__\rebel_engine.cpython-312.pyc | 02/09/2026 03:01:29
- C:\Rebel Technologies\Rebel Master\Python\rebel_engine.py | 02/09/2026 03:01:02
- C:\Rebel Technologies\Rebel Master\ML\baseline_locked.yaml | 02/09/2026 02:53:29
- C:\Rebel Technologies\Rebel Master\ML\rebel_baseline_v4.py | 02/09/2026 02:50:02
- C:\Rebel Technologies\Rebel Master\ML\rebel_baseline_v3.py | 02/09/2026 02:33:03
- C:\Rebel Technologies\Rebel Master\ML\rebel_baseline_v2.py | 02/09/2026 02:19:09
- C:\Rebel Technologies\Rebel Master\ML\rebel_baseline_v1.py | 02/09/2026 02:15:31
- C:\Rebel Technologies\Rebel Master\logs\profit_lock.txt | 02/09/2026 00:58:56

## Quick status

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
