"""
ML Label Recovery - Backfill missing labels from MT5 deal history
=================================================================
Scans MT5 closed deals and creates labels for trades that have
features but no matching labels.
"""

import os
import csv
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Set, Dict, List, Any, Optional

# Paths
TRADE_FEATURES_FILE = r"C:\Rebel Technologies\Rebel Master\ML\trade_features.csv"
LABELS_FILE = r"C:\Rebel Technologies\Rebel Master\ML\labels.csv"
CONFIG_FILE = r"C:\Rebel Technologies\Rebel Master\Config\master_config.yaml"
DEFAULT_MT5_PATH = r"C:\Program Files\Axi MetaTrader 5 Terminal\terminal64.exe"


def _load_mt5_path_from_config() -> Optional[str]:
    """Load MT5 terminal path from master_config.yaml if available."""
    try:
        import yaml
    except Exception:
        return None
    
    if not os.path.exists(CONFIG_FILE):
        return None
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("mt5") or {}).get("path")
    except Exception:
        return None


def _resolve_mt5_path(override_path: Optional[str] = None) -> str:
    """Resolve MT5 terminal path, preferring CLI override then config."""
    if override_path:
        return override_path
    cfg_path = _load_mt5_path_from_config()
    return cfg_path or DEFAULT_MT5_PATH


def get_feature_tickets() -> Dict[int, Dict]:
    """Get all tickets from trade_features.csv with their data."""
    features = {}
    if os.path.exists(TRADE_FEATURES_FILE):
        with open(TRADE_FEATURES_FILE, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ticket = int(row["ticket"])
                    features[ticket] = row
                except (ValueError, KeyError):
                    pass
    return features


def get_label_tickets() -> Set[int]:
    """Get all tickets that already have labels."""
    tickets = set()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    tickets.add(int(row["ticket"]))
                except (ValueError, KeyError):
                    pass
    return tickets


def get_closed_deals(days_back: int = 30) -> List[Any]:
    """Get closed deals from MT5 history."""
    from_date = datetime.now() - timedelta(days=days_back)
    to_date = datetime.now()
    
    deals = mt5.history_deals_get(from_date, to_date)
    if deals is None:
        return []
    
    # Filter for exit deals only (entry=1 means exit/close)
    return [d for d in deals if d.entry == 1]


def calculate_atr(symbol: str, period: int = 14) -> float:
    """Calculate ATR for a symbol."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 50)
        if rates is None or len(rates) < period + 1:
            return 0.0
        
        df = pd.DataFrame(rates)
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if len(atr) > 0 and pd.notna(atr.iloc[-1]) else 0.0
    except:
        return 0.0


def recover_missing_labels(days_back: int = 30, dry_run: bool = False, mt5_path: Optional[str] = None) -> int:
    """
    Recover missing labels from MT5 deal history.
    
    Args:
        days_back: How many days of history to scan
        dry_run: If True, only show what would be recovered without writing
        
    Returns:
        Number of labels recovered
    """
    print("=" * 60)
    print("ML LABEL RECOVERY")
    print("=" * 60)
    
    # Initialize MT5 with Rebel Master's terminal
    resolved_path = _resolve_mt5_path(mt5_path)
    print(f"[MT5] Connecting to: {resolved_path}")
    if not mt5.initialize(path=resolved_path):
        print(f"[ERROR] MT5 initialization failed: {mt5.last_error()}")
        return 0
    
    # Get existing data
    feature_data = get_feature_tickets()
    existing_labels = get_label_tickets()
    
    print(f"Trade features: {len(feature_data)}")
    print(f"Existing labels: {len(existing_labels)}")
    
    # Find features without labels
    missing_tickets = set(feature_data.keys()) - existing_labels
    print(f"Features without labels: {len(missing_tickets)}")
    
    if not missing_tickets:
        print("\n[OK] No missing labels to recover!")
        mt5.shutdown()
        return 0
    
    # Get closed deals from MT5
    closed_deals = get_closed_deals(days_back)
    print(f"Closed deals in last {days_back} days: {len(closed_deals)}")
    
    # Build deal lookup by position_id (ticket)
    deal_lookup = {}
    for deal in closed_deals:
        pos_id = deal.position_id
        if pos_id not in deal_lookup:
            deal_lookup[pos_id] = deal
    
    # Find recoverable labels
    recoverable = []
    for ticket in missing_tickets:
        if ticket in deal_lookup:
            deal = deal_lookup[ticket]
            feature = feature_data[ticket]
            recoverable.append((ticket, deal, feature))
    
    print(f"Recoverable labels: {len(recoverable)}")
    
    if not recoverable:
        print("\n[!] No matching closed deals found for missing labels")
        print("    (trades may still be open or closed before history window)")
        mt5.shutdown()
        return 0
    
    print()
    print("-" * 60)
    print("RECOVERING LABELS:")
    print("-" * 60)
    
    # Prepare labels to write
    labels_to_write = []
    
    for ticket, deal, feature in recoverable:
        symbol = deal.symbol
        entry_price = float(feature.get("entry_price", 0))
        exit_price = deal.price
        direction = feature.get("direction", "long")
        
        # Calculate P&L
        if direction == "long":
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
        
        # Get SL distance from feature
        sl = float(feature.get("sl", 0))
        sl_distance = abs(entry_price - sl) if sl > 0 else 0.0
        
        # Get ATR from feature or calculate
        atr = float(feature.get("atr", 0))
        if atr == 0:
            atr = calculate_atr(symbol)
        
        # Calculate metrics
        rr = (pnl / sl_distance) if sl_distance > 0 else 0.0
        norm_pnl = (pnl / atr) if atr > 0 else 0.0
        label = 1 if pnl > 0 else 0
        outcome_class = "win" if pnl > 0 else "loss"
        reward_ratio = rr
        vol_norm_reward = (reward_ratio / atr) if atr > 0 else 0
        
        # Time in trade (approximate from deal timestamp)
        close_time = datetime.fromtimestamp(deal.time)
        entry_time_str = feature.get("timestamp", "")
        try:
            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
            time_in_trade = (close_time - entry_time).total_seconds() / 60.0
        except:
            time_in_trade = 0.0
        
        # MFE/MAE - not available from deal history, use 0
        mfe = deal.profit if deal.profit > 0 else 0
        mae = abs(deal.profit) if deal.profit < 0 else 0
        
        label_row = {
            "timestamp": close_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "entry_price": round(entry_price, 6),
            "exit_price": round(exit_price, 6),
            "sl_distance": round(sl_distance, 6),
            "atr": round(atr, 6),
            "pnl": round(pnl, 6),
            "rr": round(rr, 4),
            "norm_pnl": round(norm_pnl, 4),
            "label": label,
            "outcome_class": outcome_class,
            "reward_ratio": round(reward_ratio, 4),
            "mfe": round(mfe, 6),
            "mae": round(mae, 6),
            "time_in_trade": round(time_in_trade, 2),
            "volatility_normalized_reward": round(vol_norm_reward, 6)
        }
        
        labels_to_write.append(label_row)
        
        status = "[WIN]" if label == 1 else "[LOSS]"
        print(f"  {status} {symbol:15} | Ticket: {ticket} | PnL: {pnl:>10.5f} | RR: {rr:>6.2f}")
    
    print("-" * 60)
    
    if dry_run:
        print(f"\n[DRY RUN] Would recover {len(labels_to_write)} labels")
        mt5.shutdown()
        return 0
    
    # Write labels to file
    if labels_to_write:
        # Get column order from existing file or use default
        columns = [
            "timestamp", "ticket", "symbol", "direction", "entry_price", "exit_price",
            "sl_distance", "atr", "pnl", "rr", "norm_pnl", "label", "outcome_class",
            "reward_ratio", "mfe", "mae", "time_in_trade", "volatility_normalized_reward"
        ]
        
        with open(LABELS_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            for row in labels_to_write:
                writer.writerow(row)
        
        print(f"\n[OK] Recovered {len(labels_to_write)} labels!")
        
        # Count wins/losses
        wins = sum(1 for r in labels_to_write if r["label"] == 1)
        losses = len(labels_to_write) - wins
        print(f"    Wins: {wins} | Losses: {losses}")
    
    mt5.shutdown()
    return len(labels_to_write)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recover missing ML labels from MT5 history")
    parser.add_argument("--days", type=int, default=30, help="Days of history to scan (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be recovered without writing")
    parser.add_argument("--mt5-path", type=str, default=None, help="Override MT5 terminal path")
    args = parser.parse_args()
    
    recovered = recover_missing_labels(days_back=args.days, dry_run=args.dry_run, mt5_path=args.mt5_path)
    
    print()
    print("=" * 60)
    if recovered > 0:
        print(f"RECOVERY COMPLETE: {recovered} labels added")
    else:
        print("NO LABELS TO RECOVER")
    print("=" * 60)
