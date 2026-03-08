"""Analyze how trades are exiting: profit lock rung, full TP, full SL, or other."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from collections import defaultdict

MT5_PATH = r"C:\MT5__TRADER\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print(f"MT5 init failed: {mt5.last_error()}")
    exit(1)

DAYS_BACK = 7
from_date = datetime.now() - timedelta(days=DAYS_BACK)
to_date = datetime.now() + timedelta(days=1)

deals = mt5.history_deals_get(from_date, to_date)
orders = mt5.history_orders_get(from_date, to_date)

entries = {}
exits = {}
for d in deals:
    if d.entry == 0:
        entries[d.position_id] = d
    elif d.entry == 1:
        exits[d.position_id] = d

# Build order lookup for SL/TP
order_info = {}
if orders:
    for o in orders:
        if o.position_id in entries:
            if o.position_id not in order_info:
                order_info[o.position_id] = {"sl": o.sl, "tp": o.tp}

closed_ids = set(entries.keys()) & set(exits.keys())

trades = []
for pid in closed_ids:
    entry = entries[pid]
    exit_d = exits[pid]
    entry_time = datetime.fromtimestamp(entry.time, tz=timezone.utc)
    exit_time = datetime.fromtimestamp(exit_d.time, tz=timezone.utc)
    entry_price = entry.price
    exit_price = exit_d.price
    profit = exit_d.profit + exit_d.swap + exit_d.commission

    direction = "BUY" if entry.type == 0 else "SELL"

    oi = order_info.get(pid, {"sl": 0, "tp": 0})
    sl = oi["sl"]
    tp = oi["tp"]

    sl_distance = abs(entry_price - sl) if sl > 0 else 0
    tp_distance = abs(tp - entry_price) if tp > 0 else 0

    # How far did price move in trade direction?
    if direction == "BUY":
        price_move = exit_price - entry_price
    else:
        price_move = entry_price - exit_price

    # R-multiple at exit
    r_at_exit = price_move / sl_distance if sl_distance > 0 else 0

    # Classify exit type
    if sl_distance > 0:
        sl_hit_tolerance = 0.15  # within 15% of SL distance
        tp_hit_tolerance = 0.15

        dist_to_sl = abs(exit_price - sl) if sl > 0 else 999
        dist_to_tp = abs(exit_price - tp) if tp > 0 else 999

        if dist_to_sl < sl_distance * sl_hit_tolerance:
            exit_type = "FULL_SL"
        elif tp > 0 and dist_to_tp < tp_distance * tp_hit_tolerance:
            exit_type = "FULL_TP"
        elif r_at_exit >= 1.5:
            exit_type = "LOCK_HIGH"
        elif r_at_exit >= 0.8:
            exit_type = "LOCK_MID"
        elif r_at_exit >= 0.3:
            exit_type = "LOCK_LOW"
        elif r_at_exit >= 0.05:
            exit_type = "LOCK_MIN"
        elif r_at_exit >= -0.1:
            exit_type = "BREAKEVEN"
        else:
            exit_type = "EARLY_CLOSE"
    else:
        exit_type = "UNKNOWN"

    hour = entry_time.hour
    if 21 <= hour or hour < 7:
        session = "TOKYO"
    elif 7 <= hour < 13:
        session = "LONDON"
    else:
        session = "NEW_YORK"

    sym = entry.symbol.upper()
    if any(k in sym for k in ("US500", "US30", "USTECH", "NAS100", "GER40", "UK100",
                               "JPN225", "AUS200", "EU50", "SPA35", "FRA40", "CN50",
                               "US2000", "HK50", "STOXX", "DJ30")):
        asset = "Indices"
    elif sym.startswith(("XAU", "XAG", "XPT", "COPPER")):
        asset = "Metals"
    elif any(k in sym for k in ("OIL", "BRENT", "WTI", "NATGAS")):
        asset = "Energy"
    else:
        asset = "FX"

    trades.append({
        "symbol": entry.symbol,
        "direction": direction,
        "entry_time": entry_time,
        "session": session,
        "asset": asset,
        "profit": profit,
        "outcome": "WIN" if profit > 0 else "LOSS",
        "r_at_exit": round(r_at_exit, 2),
        "exit_type": exit_type,
        "sl_distance": sl_distance,
        "tp_distance": tp_distance,
        "duration_min": (exit_time - entry_time).total_seconds() / 60,
    })

trades.sort(key=lambda t: t["entry_time"])

# EXIT TYPE DISTRIBUTION
print("=" * 80)
print("  HOW ARE TRADES EXITING? (Last 7 days)")
print("=" * 80)

exit_types = defaultdict(lambda: {"n": 0, "w": 0, "l": 0, "pnl": 0.0, "r_sum": 0.0})
for t in trades:
    et = exit_types[t["exit_type"]]
    et["n"] += 1
    et["pnl"] += t["profit"]
    et["r_sum"] += t["r_at_exit"]
    if t["outcome"] == "WIN":
        et["w"] += 1
    else:
        et["l"] += 1

print(f"\n  {'EXIT TYPE':14s} {'#':>5s} {'W':>4s} {'L':>4s} {'WR%':>6s} {'P/L':>10s} {'AvgR':>7s} {'AvgPnL':>9s}")
print(f"  {'-'*66}")
type_order = ["FULL_TP", "LOCK_HIGH", "LOCK_MID", "LOCK_LOW", "LOCK_MIN", "BREAKEVEN", "EARLY_CLOSE", "FULL_SL", "UNKNOWN"]
for etype in type_order:
    if etype not in exit_types:
        continue
    et = exit_types[etype]
    wr = et["w"] / et["n"] * 100 if et["n"] > 0 else 0
    avg_r = et["r_sum"] / et["n"] if et["n"] > 0 else 0
    avg_pnl = et["pnl"] / et["n"] if et["n"] > 0 else 0
    print(f"  {etype:14s} {et['n']:5d} {et['w']:4d} {et['l']:4d} {wr:5.1f}% {et['pnl']:+10.2f} {avg_r:+6.2f}R {avg_pnl:+8.2f}")

# R-MULTIPLE HISTOGRAM
print(f"\n{'='*80}")
print("  R-MULTIPLE AT EXIT (where are trades closing?)")
print("=" * 80)

r_buckets = [
    ("-1.0R or worse", lambda r: r <= -0.9),
    ("-0.5R to -0.9R", lambda r: -0.9 < r <= -0.5),
    ("-0.1R to -0.5R", lambda r: -0.5 < r <= -0.1),
    ("Breakeven", lambda r: -0.1 < r <= 0.1),
    ("+0.1R to +0.3R", lambda r: 0.1 < r <= 0.3),
    ("+0.3R to +0.5R", lambda r: 0.3 < r <= 0.5),
    ("+0.5R to +1.0R", lambda r: 0.5 < r <= 1.0),
    ("+1.0R to +1.5R", lambda r: 1.0 < r <= 1.5),
    ("+1.5R to +2.0R", lambda r: 1.5 < r <= 2.0),
    ("+2.0R or better", lambda r: r > 2.0),
]

print(f"\n  {'R BUCKET':20s} {'#':>5s} {'%':>6s}  BAR")
print(f"  {'-'*60}")
for label, check in r_buckets:
    count = sum(1 for t in trades if check(t["r_at_exit"]))
    pct = count / len(trades) * 100 if trades else 0
    bar = "#" * int(pct)
    print(f"  {label:20s} {count:5d} {pct:5.1f}%  {bar}")

# BY SESSION
print(f"\n{'='*80}")
print("  EXIT TYPES BY SESSION")
print("=" * 80)
for session in ["TOKYO", "LONDON", "NEW_YORK"]:
    st = [t for t in trades if t["session"] == session]
    if not st:
        continue
    print(f"\n  --- {session} ({len(st)} trades) ---")
    
    full_sl = sum(1 for t in st if t["exit_type"] == "FULL_SL")
    full_tp = sum(1 for t in st if t["exit_type"] == "FULL_TP")
    locks = sum(1 for t in st if t["exit_type"].startswith("LOCK"))
    be = sum(1 for t in st if t["exit_type"] == "BREAKEVEN")
    early = sum(1 for t in st if t["exit_type"] == "EARLY_CLOSE")
    
    sl_pnl = sum(t["profit"] for t in st if t["exit_type"] == "FULL_SL")
    tp_pnl = sum(t["profit"] for t in st if t["exit_type"] == "FULL_TP")
    lock_pnl = sum(t["profit"] for t in st if t["exit_type"].startswith("LOCK"))
    
    print(f"    Full SL:      {full_sl:3d} ({full_sl/len(st)*100:4.1f}%)  P/L: ${sl_pnl:+.2f}")
    print(f"    Full TP:      {full_tp:3d} ({full_tp/len(st)*100:4.1f}%)  P/L: ${tp_pnl:+.2f}")
    print(f"    Profit Lock:  {locks:3d} ({locks/len(st)*100:4.1f}%)  P/L: ${lock_pnl:+.2f}")
    print(f"    Breakeven:    {be:3d} ({be/len(st)*100:4.1f}%)")
    print(f"    Early Close:  {early:3d} ({early/len(st)*100:4.1f}%)")
    
    avg_r_w = [t["r_at_exit"] for t in st if t["outcome"] == "WIN"]
    avg_r_l = [t["r_at_exit"] for t in st if t["outcome"] == "LOSS"]
    if avg_r_w:
        print(f"    Avg R on wins:   {sum(avg_r_w)/len(avg_r_w):+.2f}R")
    if avg_r_l:
        print(f"    Avg R on losses: {sum(avg_r_l)/len(avg_r_l):+.2f}R")

# PROFIT LOCK LADDER ANALYSIS
print(f"\n{'='*80}")
print("  PROFIT LOCK IMPACT: WINS THAT LOCKED EARLY vs RAN TO TP")
print("=" * 80)

winners = [t for t in trades if t["outcome"] == "WIN"]
locked_wins = [t for t in winners if t["exit_type"].startswith("LOCK")]
tp_wins = [t for t in winners if t["exit_type"] == "FULL_TP"]

print(f"\n  Total winners: {len(winners)}")
print(f"  Locked early:  {len(locked_wins)} (avg {sum(t['r_at_exit'] for t in locked_wins)/max(len(locked_wins),1):+.2f}R, avg ${sum(t['profit'] for t in locked_wins)/max(len(locked_wins),1):+.2f})")
print(f"  Full TP hit:   {len(tp_wins)} (avg {sum(t['r_at_exit'] for t in tp_wins)/max(len(tp_wins),1):+.2f}R, avg ${sum(t['profit'] for t in tp_wins)/max(len(tp_wins),1):+.2f})")

# What if locked wins had run to 1R instead?
if locked_wins:
    actual_pnl = sum(t["profit"] for t in locked_wins)
    # Estimate: if avg R was X, and it had been 1.0R instead
    avg_locked_r = sum(t["r_at_exit"] for t in locked_wins) / len(locked_wins)
    multiplier = 1.0 / avg_locked_r if avg_locked_r > 0 else 1
    potential_pnl = actual_pnl * multiplier
    print(f"\n  WHAT IF locked wins ran to 1.0R instead:")
    print(f"    Actual P/L from locked wins:    ${actual_pnl:+.2f}")
    print(f"    Potential P/L at 1.0R:           ${potential_pnl:+.2f}")
    print(f"    Left on table:                   ${potential_pnl - actual_pnl:+.2f}")

# DURATION
print(f"\n{'='*80}")
print("  TRADE DURATION (minutes)")
print("=" * 80)

w_dur = [t["duration_min"] for t in trades if t["outcome"] == "WIN"]
l_dur = [t["duration_min"] for t in trades if t["outcome"] == "LOSS"]
print(f"  Winners avg duration:  {sum(w_dur)/max(len(w_dur),1):.0f} min")
print(f"  Losers avg duration:   {sum(l_dur)/max(len(l_dur),1):.0f} min")

for etype in ["FULL_SL", "FULL_TP", "LOCK_LOW", "LOCK_MIN"]:
    subset = [t for t in trades if t["exit_type"] == etype]
    if subset:
        avg_dur = sum(t["duration_min"] for t in subset) / len(subset)
        print(f"  {etype:14s} avg:   {avg_dur:.0f} min")

print(f"\n{'='*80}")
mt5.shutdown()
