import json
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "trades.jsonl")

trades = []
with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("outcome")]

def get_session(ts_str):
    if not ts_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        h = dt.hour
        if 0 <= h < 7:
            return "TOKYO"
        elif 7 <= h < 13:
            return "LONDON"
        elif 13 <= h < 16:
            return "OVERLAP"
        elif 16 <= h < 21:
            return "NEW_YORK"
        else:
            return "OFF_HOURS"
    except:
        return "unknown"

# Overall session stats
sessions = {}
for t in closed:
    s = get_session(t.get("close_time"))
    if s not in sessions:
        sessions[s] = {"wins": 0, "losses": 0, "profit": 0.0, "trades": 0}
    sessions[s]["trades"] += 1
    sessions[s]["profit"] += float(t.get("profit", 0))
    if t["outcome"] == "WIN":
        sessions[s]["wins"] += 1
    else:
        sessions[s]["losses"] += 1

print("=" * 65)
print(f"{'SESSION':<12} {'TRADES':>6} {'WINS':>5} {'LOSSES':>6} {'WIN%':>6} {'P/L':>10}")
print("=" * 65)
for s in ["TOKYO", "LONDON", "OVERLAP", "NEW_YORK", "OFF_HOURS", "unknown"]:
    if s in sessions:
        d = sessions[s]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(f"{s:<12} {d['trades']:>6} {d['wins']:>5} {d['losses']:>6} {wr:>5.1f}% {d['profit']:>+10.2f}")

# Per-group per-session
print()
print("=" * 65)
print(f"{'GROUP':<10} {'SESSION':<12} {'TRADES':>6} {'WINS':>5} {'LOSSES':>6} {'WIN%':>6} {'P/L':>10}")
print("=" * 65)

group_session = {}
for t in closed:
    g = t.get("group", "unknown")
    s = get_session(t.get("close_time"))
    key = (g, s)
    if key not in group_session:
        group_session[key] = {"wins": 0, "losses": 0, "profit": 0.0, "trades": 0}
    group_session[key]["trades"] += 1
    group_session[key]["profit"] += float(t.get("profit", 0))
    if t["outcome"] == "WIN":
        group_session[key]["wins"] += 1
    else:
        group_session[key]["losses"] += 1

for key in sorted(group_session, key=lambda x: group_session[x]["profit"], reverse=True):
    g, s = key
    d = group_session[key]
    wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
    print(f"{g:<10} {s:<12} {d['trades']:>6} {d['wins']:>5} {d['losses']:>6} {wr:>5.1f}% {d['profit']:>+10.2f}")

# Per-symbol per-session (top earners and worst)
print()
print("=" * 65)
print("TOP 10 SYMBOL+SESSION COMBOS")
print("=" * 65)

sym_session = {}
for t in closed:
    sym = t["symbol"]
    s = get_session(t.get("close_time"))
    key = (sym, s)
    if key not in sym_session:
        sym_session[key] = {"wins": 0, "losses": 0, "profit": 0.0, "trades": 0, "group": t.get("group", "?")}
    sym_session[key]["trades"] += 1
    sym_session[key]["profit"] += float(t.get("profit", 0))
    if t["outcome"] == "WIN":
        sym_session[key]["wins"] += 1
    else:
        sym_session[key]["losses"] += 1

print(f"{'SYMBOL':<12} {'SESSION':<12} {'TRADES':>6} {'WINS':>5} {'LOSSES':>6} {'WIN%':>6} {'P/L':>10}")
print("-" * 65)
sorted_ss = sorted(sym_session, key=lambda x: sym_session[x]["profit"], reverse=True)
for key in sorted_ss[:10]:
    sym, s = key
    d = sym_session[key]
    wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
    print(f"{sym:<12} {s:<12} {d['trades']:>6} {d['wins']:>5} {d['losses']:>6} {wr:>5.1f}% {d['profit']:>+10.2f}")

print()
print("BOTTOM 10 SYMBOL+SESSION COMBOS")
print("-" * 65)
for key in sorted_ss[-10:]:
    sym, s = key
    d = sym_session[key]
    wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
    print(f"{sym:<12} {s:<12} {d['trades']:>6} {d['wins']:>5} {d['losses']:>6} {wr:>5.1f}% {d['profit']:>+10.2f}")
