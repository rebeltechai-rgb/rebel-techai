import json
import os
from datetime import datetime

TRADES_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "trades.jsonl")

trades = []
with open(TRADES_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("outcome")]
closed.sort(key=lambda t: t.get("close_time", t.get("timestamp", "")))

total = len(closed)
wins = sum(1 for t in closed if t["outcome"] == "WIN")
losses = total - wins
wr = wins / total * 100 if total else 0

total_pnl = 0.0
peak_pnl = 0.0
max_dd = 0.0
running_pnl = 0.0
equity_curve = []

consec_wins = 0
consec_losses = 0
max_consec_wins = 0
max_consec_losses = 0

biggest_win = 0.0
biggest_loss = 0.0

total_win_pnl = 0.0
total_loss_pnl = 0.0

for t in closed:
    p = float(t.get("profit", 0))
    running_pnl += p
    equity_curve.append(running_pnl)

    if running_pnl > peak_pnl:
        peak_pnl = running_pnl
    dd = peak_pnl - running_pnl
    if dd > max_dd:
        max_dd = dd

    if t["outcome"] == "WIN":
        total_win_pnl += p
        consec_wins += 1
        consec_losses = 0
        if consec_wins > max_consec_wins:
            max_consec_wins = consec_wins
        if p > biggest_win:
            biggest_win = p
    else:
        total_loss_pnl += p
        consec_losses += 1
        consec_wins = 0
        if consec_losses > max_consec_losses:
            max_consec_losses = consec_losses
        if p < biggest_loss:
            biggest_loss = p

avg_win = total_win_pnl / wins if wins else 0
avg_loss = total_loss_pnl / losses if losses else 0
profit_factor = abs(total_win_pnl / total_loss_pnl) if total_loss_pnl != 0 else 0
expectancy = running_pnl / total if total else 0
rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

first_trade = closed[0].get("close_time", closed[0].get("timestamp", "?"))[:10] if closed else "?"
last_trade = closed[-1].get("close_time", closed[-1].get("timestamp", "?"))[:10] if closed else "?"

print("=" * 65)
print("  REBEL TRADER — RULES-ONLY PERFORMANCE REPORT")
print(f"  Period: {first_trade} to {last_trade}")
print("=" * 65)

print(f"\n  Total Closed Trades:    {total}")
print(f"  Wins:                   {wins}")
print(f"  Losses:                 {losses}")
print(f"  Win Rate:               {wr:.1f}%")

print(f"\n  Total P/L:              {running_pnl:+.2f}")
print(f"  Peak P/L:               {peak_pnl:+.2f}")
print(f"  Max Drawdown:           {max_dd:.2f}")

print(f"\n  Avg Win:                {avg_win:+.2f}")
print(f"  Avg Loss:               {avg_loss:+.2f}")
print(f"  R:R Ratio:              {rr_ratio:.2f}")
print(f"  Profit Factor:          {profit_factor:.2f}")
print(f"  Expectancy/Trade:       {expectancy:+.2f}")

print(f"\n  Biggest Win:            {biggest_win:+.2f}")
print(f"  Biggest Loss:           {biggest_loss:+.2f}")
print(f"  Max Consec Wins:        {max_consec_wins}")
print(f"  Max Consec Losses:      {max_consec_losses}")

# By group
print("\n" + "=" * 65)
print("  BY GROUP")
print("=" * 65)
print(f"  {'GROUP':<12} {'TRADES':>6} {'WINS':>5} {'LOSSES':>6} {'WR%':>6} {'P/L':>10} {'AVG_W':>8} {'AVG_L':>8}")
print("-" * 65)

groups = {}
for t in closed:
    g = t.get("group", "unknown")
    if g not in groups:
        groups[g] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0, "win_pnl": 0.0, "loss_pnl": 0.0}
    groups[g]["trades"] += 1
    p = float(t.get("profit", 0))
    groups[g]["pnl"] += p
    if t["outcome"] == "WIN":
        groups[g]["wins"] += 1
        groups[g]["win_pnl"] += p
    else:
        groups[g]["losses"] += 1
        groups[g]["loss_pnl"] += p

for g in sorted(groups, key=lambda x: groups[x]["pnl"], reverse=True):
    d = groups[g]
    wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
    aw = d["win_pnl"] / d["wins"] if d["wins"] else 0
    al = d["loss_pnl"] / d["losses"] if d["losses"] else 0
    print(f"  {g:<12} {d['trades']:>6} {d['wins']:>5} {d['losses']:>6} {wr:>5.1f}% {d['pnl']:>+10.2f} {aw:>+8.2f} {al:>+8.2f}")
