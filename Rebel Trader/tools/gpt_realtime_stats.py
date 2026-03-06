"""Quick GPT real-time stats from trades.jsonl"""
import json
from collections import Counter

TRADES_FILE = r"C:\Rebel Technologies\Rebel Trader\logs\trades.jsonl"

gpt_entries = {}
all_closes = {}

with open(TRADES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except Exception:
            continue

        ticket = t.get("ticket")
        ds = t.get("decision_source", "")
        status = t.get("status", "")

        if ds == "GPT_ASSIST" and ticket:
            gpt_entries[ticket] = t
        if status == "CLOSED" and ticket:
            all_closes[ticket] = t

# GPT-assisted trade outcomes
gpt_closed = {tk: all_closes[tk] for tk in gpt_entries if tk in all_closes}
gpt_open = {tk: gpt_entries[tk] for tk in gpt_entries if tk not in all_closes}

print("=" * 65)
print("  GPT REAL-TIME STATS (VETO PHASE)")
print("=" * 65)
print(f"  GPT_ASSIST entries:  {len(gpt_entries)}")
print(f"  Closed:              {len(gpt_closed)}")
print(f"  Still open:          {len(gpt_open)}")

if gpt_entries:
    dates = [e.get("timestamp", "")[:10] for e in gpt_entries.values()]
    print(f"  Date range:          {min(dates)} to {max(dates)}")

    rs_vals = Counter(str(e.get("risk_scale", "?")) for e in gpt_entries.values())
    print(f"\n  Risk scale applied:")
    for rs, cnt in rs_vals.most_common():
        print(f"    {rs}: {cnt} trades")

# Closed outcomes
if gpt_closed:
    wins = sum(1 for c in gpt_closed.values() if c.get("outcome") == "WIN")
    losses = len(gpt_closed) - wins
    total_pnl = sum(c.get("profit", 0) for c in gpt_closed.values())
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"\n  CLOSED OUTCOMES:")
    print(f"    Wins: {wins} | Losses: {losses} | WR: {wr:.1f}%")
    print(f"    Total P/L: ${total_pnl:+.2f}")

    # By group
    grp = {}
    for tk, close in gpt_closed.items():
        entry = gpt_entries[tk]
        g = entry.get("group", "unknown")
        if g not in grp:
            grp[g] = {"w": 0, "l": 0, "pnl": 0.0}
        grp[g]["pnl"] += close.get("profit", 0)
        if close.get("outcome") == "WIN":
            grp[g]["w"] += 1
        else:
            grp[g]["l"] += 1

    print(f"\n  BY GROUP:")
    print(f"  {'GROUP':12s} {'TRADES':>6s} {'WINS':>5s} {'LOSS':>5s} {'WR%':>6s} {'P/L':>10s}")
    print(f"  {'-'*50}")
    for g, d in sorted(grp.items(), key=lambda x: -(x[1]["w"] + x[1]["l"])):
        total = d["w"] + d["l"]
        wr = d["w"] / total * 100 if total > 0 else 0
        print(f"  {g:12s} {total:6d} {d['w']:5d} {d['l']:5d} {wr:5.1f}% {d['pnl']:+10.2f}")

# RULES-only comparison
rules_entries = {}
with open(TRADES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except Exception:
            continue
        if t.get("decision_source") == "RULES" and t.get("ticket"):
            rules_entries[t["ticket"]] = t

rules_closed = {tk: all_closes[tk] for tk in rules_entries if tk in all_closes}
if rules_closed:
    rw = sum(1 for c in rules_closed.values() if c.get("outcome") == "WIN")
    rl = len(rules_closed) - rw
    rpnl = sum(c.get("profit", 0) for c in rules_closed.values())
    rwr = rw / (rw + rl) * 100 if (rw + rl) > 0 else 0

    print(f"\n  RULES-ONLY COMPARISON:")
    print(f"    Entries: {len(rules_entries)} | Closed: {len(rules_closed)}")
    print(f"    Wins: {rw} | Losses: {rl} | WR: {rwr:.1f}%")
    print(f"    Total P/L: ${rpnl:+.2f}")

# Overall account
total_closed = len(all_closes)
total_wins = sum(1 for c in all_closes.values() if c.get("outcome") == "WIN")
total_losses = total_closed - total_wins
total_pnl_all = sum(c.get("profit", 0) for c in all_closes.values())
total_wr = total_wins / total_closed * 100 if total_closed > 0 else 0

print(f"\n  OVERALL ACCOUNT:")
print(f"    Total closed: {total_closed}")
print(f"    Wins: {total_wins} | Losses: {total_losses} | WR: {total_wr:.1f}%")
print(f"    Total P/L: ${total_pnl_all:+.2f}")
print("=" * 65)
