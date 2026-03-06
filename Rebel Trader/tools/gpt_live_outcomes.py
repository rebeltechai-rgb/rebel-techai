"""Pull real GPT veto-phase trade outcomes from MT5 deal history."""
import json
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone

TRADES_FILE = r"C:\Rebel Technologies\Rebel Trader\logs\trades.jsonl"
MT5_PATH = r"C:\MT5__TRADER\terminal64.exe"

# Get GPT_ASSIST entry tickets
gpt_entries = {}
with open(TRADES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except Exception:
            continue
        if t.get("decision_source") == "GPT_ASSIST" and t.get("ticket"):
            gpt_entries[t["ticket"]] = t

print(f"GPT_ASSIST entries in log: {len(gpt_entries)}")

# Connect to Trader's MT5 terminal
if not mt5.initialize(path=MT5_PATH):
    print(f"MT5 init failed: {mt5.last_error()}")
    print("Trying default path...")
    if not mt5.initialize():
        print(f"Default init also failed: {mt5.last_error()}")
        exit(1)

info = mt5.account_info()
print(f"MT5 Account: {info.login} | Server: {info.server}")

# Get deal history
from_date = datetime(2026, 2, 25)
to_date = datetime.now() + timedelta(days=1)
deals = mt5.history_deals_get(from_date, to_date)
print(f"Deals in range: {len(deals) if deals else 0}")

if not deals:
    mt5.shutdown()
    exit(1)

# Build position lookup: position_id -> list of deals
from collections import defaultdict
pos_deals = defaultdict(list)
for d in deals:
    pos_deals[d.position_id].append(d)

# Check each GPT ticket
print(f"\n{'='*70}")
print(f"  GPT VETO-PHASE TRADE OUTCOMES")
print(f"{'='*70}")

wins = 0
losses = 0
still_open = 0
total_pnl = 0.0
group_stats = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0.0, "open": 0})

for ticket, entry in sorted(gpt_entries.items()):
    symbol = entry.get("symbol", "?")
    direction = entry.get("direction", "?")
    group = entry.get("group", "?")
    risk_scale = entry.get("risk_scale", 1.0)

    # Look for this position in MT5 deals
    position_deals = pos_deals.get(ticket, [])
    exit_deals = [d for d in position_deals if d.entry == 1]

    if exit_deals:
        exit_deal = exit_deals[0]
        profit = exit_deal.profit + exit_deal.swap + exit_deal.commission
        outcome = "WIN" if profit > 0 else "LOSS"
        total_pnl += profit

        if profit > 0:
            wins += 1
            group_stats[group]["w"] += 1
        else:
            losses += 1
            group_stats[group]["l"] += 1
        group_stats[group]["pnl"] += profit
    else:
        # Check if position is still open
        positions = mt5.positions_get(ticket=ticket)
        if positions:
            still_open += 1
            group_stats[group]["open"] += 1
        else:
            # Might be closed with different position_id mapping
            still_open += 1
            group_stats[group]["open"] += 1

closed = wins + losses
wr = wins / closed * 100 if closed > 0 else 0

print(f"\n  Total GPT_ASSIST trades:  {len(gpt_entries)}")
print(f"  Closed:                   {closed}")
print(f"  Still open:               {still_open}")
print(f"  Wins: {wins} | Losses: {losses} | WR: {wr:.1f}%")
print(f"  Total P/L: ${total_pnl:+.2f}")

if group_stats:
    print(f"\n  BY GROUP:")
    print(f"  {'GROUP':12s} {'CLOSED':>6s} {'W':>4s} {'L':>4s} {'WR%':>6s} {'P/L':>10s} {'OPEN':>5s}")
    print(f"  {'-'*52}")
    for g in sorted(group_stats.keys()):
        d = group_stats[g]
        total = d["w"] + d["l"]
        wr = d["w"] / total * 100 if total > 0 else 0
        print(f"  {g:12s} {total:6d} {d['w']:4d} {d['l']:4d} {wr:5.1f}% {d['pnl']:+10.2f} {d['open']:5d}")

print(f"\n{'='*70}")
mt5.shutdown()
