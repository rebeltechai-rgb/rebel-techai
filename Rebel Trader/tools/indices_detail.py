"""Detailed indices breakdown for London and New York sessions."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from collections import defaultdict

MT5_PATH = r"C:\MT5__TRADER\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print(f"MT5 init failed: {mt5.last_error()}")
    exit(1)

info = mt5.account_info()

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

INDEX_SYMBOLS = {"US500", "US30", "USTECH", "NAS100", "NAS100.fs", "GER40", "UK100",
                 "JPN225", "AUS200", "EU50", "SPA35", "FRA40", "CN50", "US2000",
                 "HK50", "STOXX50", "DJ30.fs"}


def get_session(hour_utc):
    if 21 <= hour_utc or hour_utc < 7:
        return "TOKYO"
    elif 7 <= hour_utc < 13:
        return "LONDON"
    else:
        return "NEW_YORK"


trades = []
for pid in closed_ids:
    entry = entries[pid]
    exit_d = exits[pid]
    sym = entry.symbol.upper()
    if not any(k in sym for k in INDEX_SYMBOLS) and sym not in INDEX_SYMBOLS:
        continue
    entry_time = datetime.fromtimestamp(entry.time, tz=timezone.utc)
    exit_time = datetime.fromtimestamp(exit_d.time, tz=timezone.utc)
    profit = exit_d.profit + exit_d.swap + exit_d.commission
    trades.append({
        "symbol": entry.symbol,
        "direction": "BUY" if entry.type == 0 else "SELL",
        "entry_time": entry_time,
        "exit_time": exit_time,
        "date": entry_time.strftime("%Y-%m-%d"),
        "session": get_session(entry_time.hour),
        "profit": profit,
        "outcome": "WIN" if profit > 0 else "LOSS",
        "volume": entry.volume,
    })

trades.sort(key=lambda t: t["entry_time"])

print("=" * 85)
print("  INDICES BREAKDOWN — LONDON & NEW YORK (Last 7 days)")
print("=" * 85)

# Summary by session
for session in ["LONDON", "NEW_YORK"]:
    subset = [t for t in trades if t["session"] == session]
    if not subset:
        continue
    w = sum(1 for t in subset if t["outcome"] == "WIN")
    l = len(subset) - w
    pnl = sum(t["profit"] for t in subset)
    wr = w / len(subset) * 100
    avg_w = sum(t["profit"] for t in subset if t["outcome"] == "WIN") / max(w, 1)
    avg_l = sum(t["profit"] for t in subset if t["outcome"] == "LOSS") / max(l, 1)
    print(f"\n  {session}: {len(subset)} trades | {w}W {l}L | WR: {wr:.1f}% | P/L: ${pnl:+.2f} | AvgW: ${avg_w:+.2f} AvgL: ${avg_l:+.2f}")

    # By symbol
    syms = sorted(set(t["symbol"] for t in subset))
    print(f"  {'SYMBOL':14s} {'#':>4s} {'W':>3s} {'L':>3s} {'WR%':>6s} {'P/L':>10s} {'AvgW':>8s} {'AvgL':>8s}")
    print(f"  {'-'*60}")
    for sym in syms:
        sym_trades = [t for t in subset if t["symbol"] == sym]
        sw = sum(1 for t in sym_trades if t["outcome"] == "WIN")
        sl = len(sym_trades) - sw
        spnl = sum(t["profit"] for t in sym_trades)
        swr = sw / len(sym_trades) * 100
        saw = sum(t["profit"] for t in sym_trades if t["outcome"] == "WIN") / max(sw, 1)
        sal = sum(t["profit"] for t in sym_trades if t["outcome"] == "LOSS") / max(sl, 1)
        print(f"  {sym:14s} {len(sym_trades):4d} {sw:3d} {sl:3d} {swr:5.1f}% {spnl:+10.2f} {saw:+8.2f} {sal:+8.2f}")

# Every single trade (London + NY indices)
print(f"\n{'='*85}")
print("  EVERY INDEX TRADE — LONDON & NEW YORK")
print("=" * 85)
print(f"  {'DATE':10s} {'TIME':5s} {'SES':6s} {'SYMBOL':12s} {'DIR':4s} {'OUTCOME':7s} {'PROFIT':>10s}")
print(f"  {'-'*65}")

for t in trades:
    if t["session"] not in ("LONDON", "NEW_YORK"):
        continue
    date = t["entry_time"].strftime("%Y-%m-%d")
    time_str = t["entry_time"].strftime("%H:%M")
    marker = "<<<" if t["profit"] < -10 else ""
    print(f"  {date} {time_str} {t['session']:6s} {t['symbol']:12s} {t['direction']:4s} {t['outcome']:7s} {t['profit']:+10.2f}  {marker}")

# Without the 3 worst trades
print(f"\n{'='*85}")
print("  WHAT IF: REMOVE 3 WORST INDEX LOSSES")
print("=" * 85)

all_index = [t for t in trades if t["session"] in ("LONDON", "NEW_YORK")]
sorted_by_pnl = sorted(all_index, key=lambda t: t["profit"])
worst_3 = sorted_by_pnl[:3]
without_worst = [t for t in all_index if t not in worst_3]

print(f"\n  3 worst trades removed:")
for t in worst_3:
    print(f"    {t['entry_time'].strftime('%Y-%m-%d %H:%M')} {t['symbol']:12s} {t['profit']:+.2f}")

w = sum(1 for t in without_worst if t["outcome"] == "WIN")
l = len(without_worst) - w
pnl = sum(t["profit"] for t in without_worst)
wr = w / len(without_worst) * 100 if without_worst else 0
print(f"\n  Without worst 3: {len(without_worst)} trades | {w}W {l}L | WR: {wr:.1f}% | P/L: ${pnl:+.2f}")

w_all = sum(1 for t in all_index if t["outcome"] == "WIN")
l_all = len(all_index) - w_all
pnl_all = sum(t["profit"] for t in all_index)
wr_all = w_all / len(all_index) * 100 if all_index else 0
print(f"  With all trades: {len(all_index)} trades | {w_all}W {l_all}L | WR: {wr_all:.1f}% | P/L: ${pnl_all:+.2f}")
print(f"  Difference: ${pnl - pnl_all:+.2f}")

print(f"\n{'='*85}")
mt5.shutdown()
