import json
import os
from datetime import datetime, timedelta

SHADOW_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "gpt_shadow_log.jsonl")
TRADES_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "trades.jsonl")

shadow = []
with open(SHADOW_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            shadow.append(json.loads(line))

trades = []
with open(TRADES_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("outcome")]

# GPT vetoes: GPT said HOLD, rules said TRADE
vetoed_signals = [r for r in shadow if r.get("gpt_direction", "").upper() == "HOLD" and r.get("rules_direction", "").upper() != "HOLD"]

# Build lookup: symbol -> list of veto timestamps
veto_lookup = {}
for v in vetoed_signals:
    sym = v["symbol"]
    ts = v.get("ts", "")
    if not ts:
        continue
    if sym not in veto_lookup:
        veto_lookup[sym] = []
    try:
        veto_lookup[sym].append(datetime.fromisoformat(ts))
    except:
        pass

# Match trades to veto signals (within 10 min window)
matched_vetoed = []
matched_allowed = []

for trade in closed:
    symbol = trade["symbol"]
    trade_ts = trade.get("timestamp", "")
    if not trade_ts:
        continue
    try:
        trade_dt = datetime.fromisoformat(trade_ts)
    except:
        continue

    was_vetoed = False
    if symbol in veto_lookup:
        for v_dt in veto_lookup[symbol]:
            diff = abs((trade_dt - v_dt).total_seconds())
            if diff <= 600:  # within 10 minutes
                was_vetoed = True
                break

    if was_vetoed:
        matched_vetoed.append(trade)
    else:
        matched_allowed.append(trade)

# Stats
def print_stats(label, tlist):
    total = len(tlist)
    if total == 0:
        print(f"\n  {label}:")
        print(f"    No matched trades")
        return
    wins = sum(1 for t in tlist if t["outcome"] == "WIN")
    losses = total - wins
    pnl = sum(float(t.get("profit", 0)) for t in tlist)
    wr = wins / total * 100
    print(f"\n  {label}:")
    print(f"    Trades: {total}  W: {wins}  L: {losses}  WR: {wr:.1f}%")
    print(f"    P/L: {pnl:+.2f}")
    if wins:
        avg_w = sum(float(t.get("profit", 0)) for t in tlist if t["outcome"] == "WIN") / wins
        print(f"    Avg win:  {avg_w:+.2f}")
    if losses:
        avg_l = sum(float(t.get("profit", 0)) for t in tlist if t["outcome"] == "LOSS") / (total - wins)
        print(f"    Avg loss: {avg_l:+.2f}")

print("=" * 65)
print("  GPT VETO IMPACT ANALYSIS")
print(f"  Shadow log range: {shadow[0].get('ts','?')[:10]} to {shadow[-1].get('ts','?')[:10]}")
print(f"  Total veto signals: {len(vetoed_signals)}")
print("=" * 65)

print_stats("TRADES GPT WOULD HAVE BLOCKED", matched_vetoed)
print_stats("TRADES GPT WOULD HAVE ALLOWED", matched_allowed)

v_pnl = sum(float(t.get("profit", 0)) for t in matched_vetoed)
print(f"\n  NET IMPACT:")
if v_pnl < 0:
    print(f"    GPT veto would have SAVED: ${abs(v_pnl):.2f}")
elif v_pnl > 0:
    print(f"    GPT veto would have MISSED: ${v_pnl:.2f} in profits")
else:
    print(f"    No matched trades to analyze")

# Vetoed by group
if matched_vetoed:
    print("\n" + "=" * 65)
    print("  VETOED TRADES BY GROUP")
    print("=" * 65)
    print(f"  {'GROUP':<12} {'TRADES':>6} {'WINS':>5} {'LOSSES':>6} {'WR%':>6} {'P/L':>10}")
    print("-" * 65)
    groups = {}
    for t in matched_vetoed:
        g = t.get("group", "unknown")
        if g not in groups:
            groups[g] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
        groups[g]["trades"] += 1
        groups[g]["pnl"] += float(t.get("profit", 0))
        if t["outcome"] == "WIN":
            groups[g]["wins"] += 1
        else:
            groups[g]["losses"] += 1
    for g in sorted(groups, key=lambda x: groups[x]["pnl"]):
        d = groups[g]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(f"  {g:<12} {d['trades']:>6} {d['wins']:>5} {d['losses']:>6} {wr:>5.1f}% {d['pnl']:>+10.2f}")

# Vetoed by symbol
if matched_vetoed:
    print("\n" + "=" * 65)
    print("  VETOED TRADES BY SYMBOL (worst first)")
    print("=" * 65)
    print(f"  {'SYMBOL':<14} {'TRADES':>6} {'WINS':>5} {'LOSSES':>6} {'WR%':>6} {'P/L':>10}")
    print("-" * 65)
    syms = {}
    for t in matched_vetoed:
        s = t["symbol"]
        if s not in syms:
            syms[s] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
        syms[s]["trades"] += 1
        syms[s]["pnl"] += float(t.get("profit", 0))
        if t["outcome"] == "WIN":
            syms[s]["wins"] += 1
        else:
            syms[s]["losses"] += 1
    for s in sorted(syms, key=lambda x: syms[x]["pnl"]):
        d = syms[s]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(f"  {s:<14} {d['trades']:>6} {d['wins']:>5} {d['losses']:>6} {wr:>5.1f}% {d['pnl']:>+10.2f}")
