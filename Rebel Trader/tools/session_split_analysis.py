"""Analyze early NY vs late NY, and compare session combinations."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from collections import defaultdict

MT5_PATH = r"C:\MT5__TRADER\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print(f"MT5 init failed: {mt5.last_error()}")
    exit(1)

info = mt5.account_info()
print(f"Account: {info.login} | Balance: ${info.balance:.2f}")

DAYS_BACK = 7
from_date = datetime.now() - timedelta(days=DAYS_BACK)
to_date = datetime.now() + timedelta(days=1)

deals = mt5.history_deals_get(from_date, to_date)
entries = {}
exits = {}
for d in deals:
    if d.entry == 0:
        entries[d.position_id] = d
    elif d.entry == 1:
        exits[d.position_id] = d

closed_ids = set(entries.keys()) & set(exits.keys())

trades = []
for pid in closed_ids:
    entry = entries[pid]
    exit_d = exits[pid]
    entry_time = datetime.fromtimestamp(entry.time, tz=timezone.utc)
    profit = exit_d.profit + exit_d.swap + exit_d.commission

    hour = entry_time.hour
    if 21 <= hour or hour < 7:
        session = "TOKYO"
    elif 7 <= hour < 13:
        session = "LONDON"
    elif 13 <= hour < 17:
        session = "EARLY_NY"
    else:
        session = "LATE_NY"

    sym = entry.symbol.upper()
    if any(k in sym for k in ("US500", "US30", "USTECH", "NAS100", "GER40", "UK100",
                               "JPN225", "AUS200", "EU50", "SPA35", "FRA40", "CN50",
                               "US2000", "HK50", "STOXX", "DJ30")):
        asset = "Indices"
    elif sym.startswith(("XAU", "XAG", "XPT", "COPPER")):
        asset = "Metals"
    elif any(k in sym for k in ("OIL", "BRENT", "WTI", "NATGAS")):
        asset = "Energy"
    elif any(k in sym for k in ("BTC", "ETH", "LTC", "XRP", "ADA", "DOG", "SOL",
                                 "AVAX", "DOT", "UNI", "LNK", "XLM")):
        asset = "Crypto"
    else:
        asset = "FX"

    trades.append({
        "symbol": entry.symbol,
        "direction": "BUY" if entry.type == 0 else "SELL",
        "entry_time": entry_time,
        "hour": hour,
        "session": session,
        "asset": asset,
        "profit": profit,
        "outcome": "WIN" if profit > 0 else "LOSS",
    })

trades.sort(key=lambda t: t["entry_time"])


def stats(subset):
    if not subset:
        return {"n": 0, "w": 0, "l": 0, "wr": 0, "pnl": 0, "avg_w": 0, "avg_l": 0}
    w = sum(1 for t in subset if t["outcome"] == "WIN")
    l = len(subset) - w
    pnl = sum(t["profit"] for t in subset)
    wr = w / len(subset) * 100
    avg_w = sum(t["profit"] for t in subset if t["outcome"] == "WIN") / max(w, 1)
    avg_l = sum(t["profit"] for t in subset if t["outcome"] == "LOSS") / max(l, 1)
    return {"n": len(subset), "w": w, "l": l, "wr": wr, "pnl": pnl, "avg_w": avg_w, "avg_l": avg_l}


def print_row(label, s):
    if s["n"] == 0:
        return
    print(f"  {label:20s} {s['n']:4d}  {s['w']:4d}  {s['l']:4d}  {s['wr']:5.1f}%  {s['pnl']:+10.2f}  {s['avg_w']:+8.2f}  {s['avg_l']:+8.2f}")


hdr = f"  {'':20s} {'#':>4s}  {'W':>4s}  {'L':>4s}  {'WR%':>5s}  {'P/L':>10s}  {'AvgW':>8s}  {'AvgL':>8s}"
sep = f"  {'-'*78}"

# EARLY NY vs LATE NY
print(f"\n{'='*82}")
print("  NY SPLIT: EARLY (13:00-17:00) vs LATE (17:00-21:00)")
print("=" * 82)
print(hdr)
print(sep)
for session in ["EARLY_NY", "LATE_NY"]:
    s = stats([t for t in trades if t["session"] == session])
    print_row(session, s)
print(sep)
print_row("ALL NY", stats([t for t in trades if t["session"] in ("EARLY_NY", "LATE_NY")]))

# Early NY by asset class
print(f"\n  EARLY NY by asset:")
print(hdr)
print(sep)
for asset in sorted(set(t["asset"] for t in trades)):
    s = stats([t for t in trades if t["session"] == "EARLY_NY" and t["asset"] == asset])
    if s["n"] > 0:
        print_row(f"  {asset}", s)

print(f"\n  LATE NY by asset:")
print(hdr)
print(sep)
for asset in sorted(set(t["asset"] for t in trades)):
    s = stats([t for t in trades if t["session"] == "LATE_NY" and t["asset"] == asset])
    if s["n"] > 0:
        print_row(f"  {asset}", s)

# SESSION COMBINATIONS - what would the P/L be
print(f"\n{'='*82}")
print("  SESSION SCENARIOS: WHAT WOULD EACH COMBINATION PRODUCE?")
print("=" * 82)
print(hdr)
print(sep)

combos = [
    ("LONDON only", ["LONDON"]),
    ("TOKYO only", ["TOKYO"]),
    ("EARLY_NY only", ["EARLY_NY"]),
    ("LONDON + EARLY_NY", ["LONDON", "EARLY_NY"]),
    ("LONDON + TOKYO", ["LONDON", "TOKYO"]),
    ("TOKYO + LONDON + EARLY", ["TOKYO", "LONDON", "EARLY_NY"]),
    ("ALL (current)", ["TOKYO", "LONDON", "EARLY_NY", "LATE_NY"]),
]

for label, sessions in combos:
    s = stats([t for t in trades if t["session"] in sessions])
    print_row(label, s)

# Per-hour breakdown (all trades)
print(f"\n{'='*82}")
print("  HOURLY BREAKDOWN (UTC)")
print("=" * 82)
print(hdr)
print(sep)
for h in range(24):
    s = stats([t for t in trades if t["hour"] == h])
    if s["n"] > 0:
        session_tag = ""
        if 21 <= h or h < 7:
            session_tag = "TOKYO"
        elif 7 <= h < 13:
            session_tag = "LONDON"
        elif 13 <= h < 17:
            session_tag = "EARLY_NY"
        else:
            session_tag = "LATE_NY"
        print_row(f"{h:02d}:00 ({session_tag})", s)

print(f"\n{'='*82}")
mt5.shutdown()
