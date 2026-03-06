"""Trader session stats from MT5 deal history over the last few days."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from collections import defaultdict

MT5_PATH = r"C:\MT5__TRADER\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print(f"MT5 init failed: {mt5.last_error()}")
    exit(1)

info = mt5.account_info()
print(f"Account: {info.login} | Balance: ${info.balance:.2f} | Equity: ${info.equity:.2f}")

DAYS_BACK = 7
from_date = datetime.now() - timedelta(days=DAYS_BACK)
to_date = datetime.now() + timedelta(days=1)

deals = mt5.history_deals_get(from_date, to_date)
if not deals:
    print("No deals found")
    mt5.shutdown()
    exit(1)

# Pair entry and exit deals by position_id
entries = {}
exits = {}
for d in deals:
    if d.entry == 0:
        entries[d.position_id] = d
    elif d.entry == 1:
        exits[d.position_id] = d

closed_ids = set(entries.keys()) & set(exits.keys())
print(f"Closed trades (last {DAYS_BACK} days): {len(closed_ids)}\n")


def get_session(hour_utc):
    # Trader config: Tokyo 21:00-07:00, London 07:00-13:00, New York 13:00-21:00
    if 21 <= hour_utc or hour_utc < 7:
        return "TOKYO"
    elif 7 <= hour_utc < 13:
        return "LONDON"
    else:
        return "NEW_YORK"


def get_asset_class(symbol):
    sym = symbol.upper()
    fx_majors = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"}
    if sym in fx_majors:
        return "FX_Major"
    if sym.startswith(("XAU", "XAG", "XPT", "COPPER")):
        return "Metals"
    if any(k in sym for k in ("US500", "US30", "USTECH", "NAS100", "GER40", "UK100",
                               "JPN225", "AUS200", "EU50", "SPA35", "FRA40", "CN50",
                               "US2000", "HK50", "STOXX")):
        return "Indices"
    if any(k in sym for k in ("OIL", "BRENT", "WTI", "NATGAS")):
        return "Energy"
    if any(k in sym for k in ("BTC", "ETH", "LTC", "XRP", "ADA", "DOG", "SOL",
                               "AVAX", "DOT", "UNI", "LNK", "XLM")):
        return "Crypto"
    return "FX_Cross"


# Build trade records
trades = []
for pid in closed_ids:
    entry = entries[pid]
    exit_d = exits[pid]
    entry_time = datetime.fromtimestamp(entry.time, tz=timezone.utc)
    profit = exit_d.profit + exit_d.swap + exit_d.commission
    trades.append({
        "symbol": entry.symbol,
        "direction": "BUY" if entry.type == 0 else "SELL",
        "entry_time": entry_time,
        "date": entry_time.strftime("%Y-%m-%d"),
        "hour_utc": entry_time.hour,
        "session": get_session(entry_time.hour),
        "asset_class": get_asset_class(entry.symbol),
        "profit": profit,
        "outcome": "WIN" if profit > 0 else "LOSS",
    })

trades.sort(key=lambda t: t["entry_time"])


def print_stats(label, subset):
    if not subset:
        print(f"  {label:16s}   --  no trades  --")
        return
    w = sum(1 for t in subset if t["outcome"] == "WIN")
    l = len(subset) - w
    pnl = sum(t["profit"] for t in subset)
    wr = w / len(subset) * 100
    avg_win = sum(t["profit"] for t in subset if t["outcome"] == "WIN") / max(w, 1)
    avg_loss = sum(t["profit"] for t in subset if t["outcome"] == "LOSS") / max(l, 1)
    print(f"  {label:16s}  {len(subset):4d}  {w:4d}  {l:4d}  {wr:5.1f}%  {pnl:+10.2f}  {avg_win:+8.2f}  {avg_loss:+8.2f}")


header = f"  {'':16s}  {'#':>4s}  {'W':>4s}  {'L':>4s}  {'WR%':>5s}  {'P/L':>10s}  {'AvgW':>8s}  {'AvgL':>8s}"
sep = f"  {'-'*76}"

# BY SESSION
print("=" * 80)
print("  BY SESSION")
print("=" * 80)
print(header)
print(sep)
for session in ["TOKYO", "LONDON", "NEW_YORK"]:
    subset = [t for t in trades if t["session"] == session]
    print_stats(session, subset)
print(sep)
print_stats("ALL", trades)

# BY DAY
print(f"\n{'='*80}")
print("  BY DAY")
print("=" * 80)
print(header)
print(sep)
dates = sorted(set(t["date"] for t in trades))
for date in dates:
    subset = [t for t in trades if t["date"] == date]
    weekday = datetime.strptime(date, "%Y-%m-%d").strftime("%a")
    print_stats(f"{date} {weekday}", subset)
print(sep)
print_stats("ALL", trades)

# BY SESSION x DAY
print(f"\n{'='*80}")
print("  BY SESSION x DAY")
print("=" * 80)
print(header)
print(sep)
for date in dates:
    weekday = datetime.strptime(date, "%Y-%m-%d").strftime("%a")
    print(f"  --- {date} ({weekday}) ---")
    for session in ["TOKYO", "LONDON", "NEW_YORK"]:
        subset = [t for t in trades if t["date"] == date and t["session"] == session]
        if subset:
            print_stats(f"  {session}", subset)

# BY ASSET CLASS
print(f"\n{'='*80}")
print("  BY ASSET CLASS")
print("=" * 80)
print(header)
print(sep)
classes = sorted(set(t["asset_class"] for t in trades))
for ac in classes:
    subset = [t for t in trades if t["asset_class"] == ac]
    print_stats(ac, subset)
print(sep)
print_stats("ALL", trades)

# BY ASSET CLASS x SESSION
print(f"\n{'='*80}")
print("  BY ASSET CLASS x SESSION")
print("=" * 80)
print(header)
print(sep)
for ac in classes:
    print(f"  --- {ac} ---")
    for session in ["TOKYO", "LONDON", "NEW_YORK"]:
        subset = [t for t in trades if t["asset_class"] == ac and t["session"] == session]
        if subset:
            print_stats(f"  {session}", subset)

print(f"\n{'='*80}")
mt5.shutdown()
