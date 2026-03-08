import json
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "trades.jsonl")

trades = []
with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("outcome")]

stats = {}
for t in closed:
    sym = t["symbol"]
    if sym not in stats:
        stats[sym] = {"wins": 0, "losses": 0, "total_profit": 0.0, "trades": 0, "group": t.get("group", "?")}
    stats[sym]["trades"] += 1
    stats[sym]["total_profit"] += float(t.get("profit", 0))
    if t["outcome"] == "WIN":
        stats[sym]["wins"] += 1
    else:
        stats[sym]["losses"] += 1

groups = {}
for t in closed:
    g = t.get("group", "unknown")
    if g not in groups:
        groups[g] = {"wins": 0, "losses": 0, "total_profit": 0.0, "trades": 0}
    groups[g]["trades"] += 1
    groups[g]["total_profit"] += float(t.get("profit", 0))
    if t["outcome"] == "WIN":
        groups[g]["wins"] += 1
    else:
        groups[g]["losses"] += 1

total_profit = sum(s["total_profit"] for s in stats.values())
total_wins = sum(s["wins"] for s in stats.values())
total_losses = sum(s["losses"] for s in stats.values())
wr = total_wins / len(closed) * 100 if closed else 0

print(f"Total closed: {len(closed)}")
print(f"Overall: {total_wins}W / {total_losses}L ({wr:.1f}%) | P/L: ${total_profit:.2f}")
print()

print("=" * 70)
print(f"{'GROUP':<12} {'TRADES':>6} {'WINS':>5} {'LOSSES':>6} {'WIN%':>6} {'P/L ($)':>10}")
print("=" * 70)
for g in sorted(groups, key=lambda x: groups[x]["total_profit"], reverse=True):
    s = groups[g]
    w = s["wins"] / s["trades"] * 100 if s["trades"] else 0
    print(f"{g:<12} {s['trades']:>6} {s['wins']:>5} {s['losses']:>6} {w:>5.1f}% {s['total_profit']:>+10.2f}")

print()
print("=" * 70)
print(f"{'SYMBOL':<12} {'GRP':<8} {'TRADES':>6} {'WINS':>5} {'LOSSES':>6} {'WIN%':>6} {'P/L ($)':>10}")
print("=" * 70)
for sym in sorted(stats, key=lambda x: stats[x]["total_profit"], reverse=True):
    s = stats[sym]
    w = s["wins"] / s["trades"] * 100 if s["trades"] else 0
    print(f"{sym:<12} {s['group']:<8} {s['trades']:>6} {s['wins']:>5} {s['losses']:>6} {w:>5.1f}% {s['total_profit']:>+10.2f}")

# Bottom 10 worst performers
print()
print("=" * 70)
print("BOTTOM 10 (worst P/L)")
print("=" * 70)
bottom = sorted(stats.items(), key=lambda x: x[1]["total_profit"])[:10]
for sym, s in bottom:
    w = s["wins"] / s["trades"] * 100 if s["trades"] else 0
    print(f"{sym:<12} {s['group']:<8} {s['trades']:>6} {s['wins']:>5} {s['losses']:>6} {w:>5.1f}% {s['total_profit']:>+10.2f}")
