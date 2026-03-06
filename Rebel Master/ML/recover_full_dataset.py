"""
Recover Full ML Training Dataset from MT5 Deal History
=======================================================
Pulls all closed trades for magic 20251202, reconstructs entry-time
features from historical H1 candles, and outputs trade_features.csv
and labels.csv ready for merge_features_labels.py and baseline analysis.

Usage:
    python recover_full_dataset.py                     # Full recovery
    python recover_full_dataset.py --days 180          # Custom lookback
    python recover_full_dataset.py --dry-run            # Preview only
    python recover_full_dataset.py --merge-and-baseline # Also run merge + baseline
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_PATH = r"C:\Rebel Technologies\Rebel Master\ML"
FEATURES_FILE = os.path.join(BASE_PATH, "trade_features.csv")
LABELS_FILE = os.path.join(BASE_PATH, "labels.csv")
CONFIG_FILE = r"C:\Rebel Technologies\Rebel Master\Config\master_config.yaml"
DEFAULT_MT5_PATH = r"C:\Master Pro\terminal64.exe"

MAGIC_NUMBER = 20251202
BARS_NEEDED = 500  # enough for EMA50 + warmup

# ─── Indicator Parameters (match rebel_signals.py) ───────────────────────
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ─── Asset Classification (match rebel_signals.py) ───────────────────────
FX_MAJORS = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"}
FX_MINORS = {
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
    "NZDJPY", "NZDCAD", "NZDCHF", "CADJPY", "CADCHF", "CHFJPY"
}


def get_mt5_path() -> str:
    try:
        import yaml
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            path = (cfg.get("mt5") or {}).get("path")
            if path and os.path.exists(path):
                return path
    except Exception:
        pass
    return DEFAULT_MT5_PATH


def get_asset_class(symbol: str) -> str:
    sym = symbol.upper().replace(".A", "").replace(".M", "").replace(".FS", "").replace(".SA", "")
    if sym in FX_MAJORS:
        return "fx_major"
    if sym in FX_MINORS:
        return "fx_minor"
    crypto_prefixes = ("BTC", "ETH", "LTC", "XRP", "ADA", "DOG", "DOT", "SOL",
                       "AVAX", "LNK", "UNI", "BCH", "XLM", "AAVE", "SUSHI")
    if any(c in sym for c in crypto_prefixes) or sym.endswith(("-USD", "-JPY")):
        return "crypto"
    if sym.startswith(("XAU", "XAG", "XPT", "GOLD", "SILVER", "COPPER")):
        return "metals"
    energy_kw = ("BRENT", "WTI", "OIL", "UKOIL", "USOIL", "CRUDE", "NATGAS", "GAS")
    if any(k in sym for k in energy_kw):
        return "energies"
    soft_kw = ("COCOA", "COFFEE", "SUGAR", "COTTON", "WHEAT", "CORN", "SOY", "BEAN", "SOYBEAN")
    if any(k in sym for k in soft_kw):
        return "softs"
    if sym.startswith("USD") or sym.endswith("USD"):
        return "fx_exotic"
    indices_kw = ("US500", "US30", "USTECH", "GER40", "UK100", "JPN225", "AUS200",
                  "FRA40", "ESP35", "HK50", "EU50", "STOXX")
    if any(k in sym for k in indices_kw):
        return "indices"
    return "fx_exotic"


def get_session(hour_utc: int) -> str:
    if 0 <= hour_utc < 7:
        return "asia"
    elif 7 <= hour_utc < 13:
        return "london"
    elif 13 <= hour_utc < 21:
        return "ny"
    else:
        return "post"


# ─── Indicator Calculations (match rebel_signals.py exactly) ─────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=high.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm = pd.Series(0.0, index=high.index)
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]

    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    di_plus = 100 * (smooth_plus / atr.replace(0, 1e-10))
    di_minus = 100 * (smooth_minus / atr.replace(0, 1e-10))

    di_sum = di_plus + di_minus
    di_diff = (di_plus - di_minus).abs()
    dx = 100 * (di_diff / di_sum.replace(0, 1e-10))
    return dx.rolling(window=period).mean()


def calc_macd_hist(close: pd.Series) -> float:
    ema12 = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema26 = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = macd_line - signal_line
    return float(hist.iloc[-1]) if len(hist) > 0 else 0.0


def fetch_h1_at_time(symbol: str, entry_time: datetime) -> Optional[pd.DataFrame]:
    """Fetch H1 candles ending at or just before the trade entry time."""
    utc_time = entry_time.replace(tzinfo=None)
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, utc_time, BARS_NEEDED)
    if rates is None or len(rates) < 100:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def compute_features_at_entry(symbol: str, entry_time: datetime, entry_price: float,
                               sl: float, tp: float) -> Optional[Dict[str, Any]]:
    """Reconstruct the full feature set from H1 candles at trade entry time."""
    df = fetch_h1_at_time(symbol, entry_time)
    if df is None:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema9 = calc_ema(close, EMA_FAST)
    ema21 = calc_ema(close, EMA_MID)
    ema50 = calc_ema(close, EMA_SLOW)
    rsi = calc_rsi(close, RSI_PERIOD)
    atr = calc_atr(high, low, close, ATR_PERIOD)
    adx = calc_adx(high, low, close, ADX_PERIOD)
    macd_hist = calc_macd_hist(close)

    ema9_last = float(ema9.iloc[-1])
    ema21_last = float(ema21.iloc[-1])
    ema50_last = float(ema50.iloc[-1])
    rsi_last = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0
    atr_last = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
    adx_last = float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 0.0

    # Trend bias from EMA50 position
    if entry_price > ema50_last * 1.001:
        trend_bias = "bullish"
    elif entry_price < ema50_last * 0.999:
        trend_bias = "bearish"
    else:
        trend_bias = "neutral"

    # Volatility regime from ATR
    atr_median = float(atr.dropna().median()) if len(atr.dropna()) > 0 else atr_last
    if atr_last > atr_median * 1.5:
        vol_regime = "high"
    elif atr_last < atr_median * 0.5:
        vol_regime = "low"
    else:
        vol_regime = "normal"

    # Spread ratio from tick (best effort)
    spread_ratio = 0.0
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick and tick.ask > 0 and atr_last > 0:
            spread_ratio = (tick.ask - tick.bid) / atr_last
    except Exception:
        pass

    session = get_session(entry_time.hour)

    # Signal score approximation (EMA cross + trend + RSI + ATR + ADX)
    score = 0
    if ema9_last > ema21_last:
        score += 1
    if entry_price > ema50_last:
        score += 1
    if 30 < rsi_last < 70:
        score += 1
    if atr_last > 0:
        score += 1
    if adx_last > 25:
        score += 1

    return {
        "ema_fast": round(ema9_last, 6),
        "ema_slow": round(ema21_last, 6),
        "rsi": round(rsi_last, 2),
        "atr": round(atr_last, 6),
        "adx": round(adx_last, 2),
        "macd_hist": round(macd_hist, 6),
        "trend_bias": trend_bias,
        "volatility_regime": vol_regime,
        "spread_ratio": round(spread_ratio, 6),
        "session_state": session,
        "raw_signal": "recovered",
        "signal_score": score,
        "reason": "recovered_from_mt5",
        "score": score,
    }


# ─── Main Recovery ────────────────────────────────────────────────────────

def recover_dataset(days_back: int = 365, dry_run: bool = False, last_n: int = 0) -> Tuple[int, int]:
    """
    Pull all closed trades from MT5 and rebuild trade_features.csv + labels.csv.

    Args:
        days_back: How many days of history to scan
        dry_run: Preview only, don't write
        last_n: If > 0, only process the N most recent closed trades

    Returns (features_written, labels_written).
    """
    print("=" * 70)
    print("  FULL ML DATASET RECOVERY FROM MT5")
    print("=" * 70)

    mt5_path = get_mt5_path()
    print(f"[MT5] Connecting to: {mt5_path}")
    if not mt5.initialize(path=mt5_path):
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        return (0, 0)

    account = mt5.account_info()
    if account:
        print(f"[MT5] Account: {account.login} | Server: {account.server}")

    # Pull ALL deals for the lookback period
    from_date = datetime.now() - timedelta(days=days_back)
    to_date = datetime.now() + timedelta(days=1)

    all_deals = mt5.history_deals_get(from_date, to_date)
    if all_deals is None or len(all_deals) == 0:
        print("[ERROR] No deals found in MT5 history")
        mt5.shutdown()
        return (0, 0)

    print(f"[MT5] Total deals in history: {len(all_deals)}")

    # Filter by magic number
    our_deals = [d for d in all_deals if d.magic == MAGIC_NUMBER]
    print(f"[MT5] Deals with magic {MAGIC_NUMBER}: {len(our_deals)}")

    # Separate entry and exit deals
    entries = {}  # position_id -> deal
    exits = {}    # position_id -> deal

    for d in our_deals:
        if d.entry == 0:  # DEAL_ENTRY_IN
            entries[d.position_id] = d
        elif d.entry == 1:  # DEAL_ENTRY_OUT
            exits[d.position_id] = d

    print(f"[MT5] Entry deals: {len(entries)}")
    print(f"[MT5] Exit deals: {len(exits)}")

    # Find closed trades (have both entry and exit)
    closed_position_ids = set(entries.keys()) & set(exits.keys())
    print(f"[MT5] Closed trades (matched entry+exit): {len(closed_position_ids)}")

    if not closed_position_ids:
        print("[ERROR] No closed trades found")
        mt5.shutdown()
        return (0, 0)

    # Try to get SL/TP from order history
    print("[MT5] Fetching order history for SL/TP...")
    all_orders = mt5.history_orders_get(from_date, to_date)
    order_sl_tp = {}
    if all_orders:
        for order in all_orders:
            if order.position_id in closed_position_ids:
                if order.position_id not in order_sl_tp:
                    order_sl_tp[order.position_id] = {
                        "sl": order.sl if hasattr(order, "sl") else 0.0,
                        "tp": order.tp if hasattr(order, "tp") else 0.0,
                    }
    print(f"[MT5] Orders with SL/TP found: {len(order_sl_tp)}")

    # Sort by entry time
    sorted_positions = sorted(closed_position_ids, key=lambda pid: entries[pid].time)

    if last_n > 0 and last_n < len(sorted_positions):
        sorted_positions = sorted_positions[-last_n:]
        print(f"[FILTER] Using last {last_n} trades only")

    print(f"\n{'-' * 70}")
    print(f"  RECONSTRUCTING FEATURES FOR {len(sorted_positions)} TRADES")
    print(f"{'-' * 70}")

    if dry_run:
        print(f"\n[DRY RUN] Would process {len(sorted_positions)} trades")
        # Show date range
        first_time = datetime.fromtimestamp(entries[sorted_positions[0]].time, tz=timezone.utc)
        last_time = datetime.fromtimestamp(entries[sorted_positions[-1]].time, tz=timezone.utc)
        print(f"  Date range: {first_time:%Y-%m-%d} to {last_time:%Y-%m-%d}")
        # Show symbol distribution
        symbols = {}
        for pid in sorted_positions:
            sym = entries[pid].symbol
            symbols[sym] = symbols.get(sym, 0) + 1
        print(f"  Unique symbols: {len(symbols)}")
        for sym, count in sorted(symbols.items(), key=lambda x: -x[1])[:15]:
            print(f"    {sym:15s}: {count}")
        mt5.shutdown()
        return (0, 0)

    # Prepare output files
    feature_columns = [
        "timestamp", "ticket", "symbol", "direction", "entry_price", "sl", "tp",
        "volume", "score", "ema_fast", "ema_slow", "rsi", "atr", "adx",
        "macd_hist", "trend_bias", "volatility_regime", "spread_ratio",
        "session_state", "raw_signal", "signal_score", "reason"
    ]
    label_columns = [
        "timestamp", "ticket", "symbol", "direction", "entry_price", "exit_price",
        "sl_distance", "atr", "pnl", "rr", "norm_pnl", "label", "outcome_class",
        "reward_ratio", "mfe", "mae", "time_in_trade", "volatility_normalized_reward"
    ]

    features_written = 0
    labels_written = 0
    errors = 0
    wins = 0
    losses = 0

    # Cache: avoid re-fetching candles for same symbol within same hour
    candle_cache: Dict[str, pd.DataFrame] = {}

    with open(FEATURES_FILE, "w", newline="") as ff, \
         open(LABELS_FILE, "w", newline="") as lf:

        feat_writer = csv.DictWriter(ff, fieldnames=feature_columns)
        label_writer = csv.DictWriter(lf, fieldnames=label_columns)
        feat_writer.writeheader()
        label_writer.writeheader()

        for i, pid in enumerate(sorted_positions):
            entry_deal = entries[pid]
            exit_deal = exits[pid]
            symbol = entry_deal.symbol

            entry_time = datetime.fromtimestamp(entry_deal.time, tz=timezone.utc)
            exit_time = datetime.fromtimestamp(exit_deal.time, tz=timezone.utc)
            entry_price = entry_deal.price
            exit_price = exit_deal.price
            volume = entry_deal.volume

            # Direction from deal type
            direction = "long" if entry_deal.type == 0 else "short"  # 0=BUY, 1=SELL

            # SL/TP from order history
            sl_tp = order_sl_tp.get(pid, {"sl": 0.0, "tp": 0.0})
            sl = sl_tp["sl"]
            tp = sl_tp["tp"]

            # Reconstruct features from historical candles
            features = compute_features_at_entry(symbol, entry_time, entry_price, sl, tp)
            if features is None:
                errors += 1
                if errors <= 10:
                    print(f"  [SKIP] {symbol:15s} ticket={pid} — no candle data at {entry_time:%Y-%m-%d %H:%M}")
                continue

            atr_val = features["atr"]
            sl_distance = abs(entry_price - sl) if sl > 0 else atr_val

            # Calculate P&L
            if direction == "long":
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            rr = (pnl / sl_distance) if sl_distance > 0 else 0.0
            norm_pnl = (pnl / atr_val) if atr_val > 0 else 0.0
            label = 1 if pnl > 0 else 0
            outcome_class = "win" if pnl > 0 else "loss"
            time_in_trade = (exit_time - entry_time).total_seconds() / 60.0
            vol_norm_reward = (rr / atr_val) if atr_val > 0 else 0.0

            if label == 1:
                wins += 1
            else:
                losses += 1

            # Write feature row
            feat_row = {
                "timestamp": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                "ticket": pid,
                "symbol": symbol,
                "direction": direction,
                "entry_price": round(entry_price, 6),
                "sl": round(sl, 6),
                "tp": round(tp, 6),
                "volume": round(volume, 4),
                "score": features["score"],
                "ema_fast": features["ema_fast"],
                "ema_slow": features["ema_slow"],
                "rsi": features["rsi"],
                "atr": features["atr"],
                "adx": features["adx"],
                "macd_hist": features["macd_hist"],
                "trend_bias": features["trend_bias"],
                "volatility_regime": features["volatility_regime"],
                "spread_ratio": features["spread_ratio"],
                "session_state": features["session_state"],
                "raw_signal": features["raw_signal"],
                "signal_score": features["signal_score"],
                "reason": features["reason"],
            }
            feat_writer.writerow(feat_row)
            features_written += 1

            # Write label row
            label_row = {
                "timestamp": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "ticket": pid,
                "symbol": symbol,
                "direction": direction,
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_price, 6),
                "sl_distance": round(sl_distance, 6),
                "atr": round(atr_val, 6),
                "pnl": round(pnl, 6),
                "rr": round(rr, 4),
                "norm_pnl": round(norm_pnl, 4),
                "label": label,
                "outcome_class": outcome_class,
                "reward_ratio": round(rr, 4),
                "mfe": 0.0,
                "mae": 0.0,
                "time_in_trade": round(time_in_trade, 2),
                "volatility_normalized_reward": round(vol_norm_reward, 6),
            }
            label_writer.writerow(label_row)
            labels_written += 1

            # Progress every 50 trades
            if (i + 1) % 50 == 0 or i == len(sorted_positions) - 1:
                elapsed_sym = f"{symbol:15s}" if i < len(sorted_positions) - 1 else "DONE"
                print(f"  [{i+1:4d}/{len(sorted_positions)}] {elapsed_sym} | "
                      f"W:{wins} L:{losses} | Errors:{errors}")

    mt5.shutdown()

    # Summary
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  RECOVERY COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Features written: {features_written}")
    print(f"  Labels written:   {labels_written}")
    print(f"  Errors/skipped:   {errors}")
    print(f"  Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")
    print(f"\n  Output files:")
    print(f"    {FEATURES_FILE}")
    print(f"    {LABELS_FILE}")
    print(f"{'=' * 70}")

    return (features_written, labels_written)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recover full ML dataset from MT5 deal history")
    parser.add_argument("--days", type=int, default=365, help="Days of history to scan (default: 365)")
    parser.add_argument("--dry-run", action="store_true", help="Preview trades without writing files")
    parser.add_argument("--last-n", type=int, default=0, help="Only process the N most recent trades (default: all)")
    parser.add_argument("--merge-and-baseline", action="store_true",
                        help="After recovery, run merge + baseline automatically")
    args = parser.parse_args()

    feats, labels = recover_dataset(days_back=args.days, dry_run=args.dry_run, last_n=args.last_n)

    if args.merge_and_baseline and feats > 0:
        print("\n\n")
        print("=" * 70)
        print("  RUNNING MERGE + BASELINE")
        print("=" * 70)

        # Run merge
        print("\n[STEP 1] Running merge_features_labels.py...")
        merge_script = os.path.join(BASE_PATH, "merge_features_labels.py")
        os.system(f'python "{merge_script}"')

        # Run baseline
        print("\n[STEP 2] Running rebel_baseline_v4.py...")
        baseline_script = os.path.join(BASE_PATH, "rebel_baseline_v4.py")
        os.system(f'python "{baseline_script}"')
