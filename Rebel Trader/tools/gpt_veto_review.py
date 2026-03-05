"""
GPT Veto Phase Review
Shows what GPT actually did since veto went live (governed_count >= 1250).
"""
import json
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
SHADOW_PATH = os.path.join(LOG_DIR, "gpt_shadow_log.jsonl")
TRADES_PATH = os.path.join(LOG_DIR, "trades.jsonl")

VETO_START = 1250

shadow = []
with open(SHADOW_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if int(rec.get("governed_count", 0) or 0) >= VETO_START:
            shadow.append(rec)

trades = []
with open(TRADES_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        trades.append(json.loads(line))

closed_trades = [t for t in trades if t.get("status") == "CLOSED" or t.get("direction") == "CLOSE"]
open_trades = [t for t in trades if t.get("direction") not in ("CLOSE",) and t.get("status") != "CLOSED"]

veto_phase_trades = []
for t in trades:
    tags = t.get("tags", {}) or {}
    ds = t.get("decision_source") or tags.get("decision_source", "")
    if ds in ("GPT_ASSIST", "GPT_PRIMARY"):
        veto_phase_trades.append(t)

total_shadow = len(shadow)
gpt_hold = [r for r in shadow if r.get("gpt_direction", "").upper() == "HOLD"]
gpt_trade = [r for r in shadow if r.get("gpt_direction", "").upper() != "HOLD"]
rules_trade = [r for r in shadow if r.get("rules_direction", "").upper() != "HOLD"]
rules_hold = [r for r in shadow if r.get("rules_direction", "").upper() == "HOLD"]

vetoed = [r for r in shadow if r.get("gpt_direction", "").upper() == "HOLD"
          and r.get("rules_direction", "").upper() != "HOLD"]
allowed = [r for r in shadow if r.get("gpt_direction", "").upper() != "HOLD"
           and r.get("rules_direction", "").upper() != "HOLD"]

print("=" * 70)
print("  GPT VETO PHASE REVIEW (governed_count >= 1250)")
print("=" * 70)

if shadow:
    ts_min = shadow[0].get("ts", "?")[:19]
    ts_max = shadow[-1].get("ts", "?")[:19]
    print(f"  Period: {ts_min} to {ts_max}")

print(f"  Total signals evaluated by GPT: {total_shadow}")
print(f"  GPT said HOLD (veto):  {len(gpt_hold)}")
print(f"  GPT said TRADE:        {len(gpt_trade)}")
print(f"  Rules wanted to trade: {len(rules_trade)}")

print(f"\n  ACTUAL VETOES (Rules wanted trade, GPT blocked): {len(vetoed)}")
print(f"  ALLOWED trades (both agreed to trade):           {len(allowed)}")

if total_shadow:
    print(f"  Veto rate on tradeable signals: {len(vetoed)}/{len(rules_trade)} "
          f"({len(vetoed)/max(len(rules_trade),1)*100:.1f}%)")

# Match vetoed signals to closed trades to see what would have happened
print("\n" + "=" * 70)
print("  VETOED SIGNALS — WHAT WOULD HAVE HAPPENED")
print("=" * 70)

vetoed_match_wins = 0
vetoed_match_losses = 0
vetoed_match_pl = 0.0
vetoed_unmatched = 0

for v in vetoed:
    sym = v.get("symbol", "")
    v_ts = v.get("ts", "")
    try:
        v_dt = datetime.fromisoformat(v_ts)
    except Exception:
        continue

    best_match = None
    best_delta = 9999999
    for t in closed_trades:
        if (t.get("symbol") or "") != sym:
            continue
        t_ts = t.get("timestamp", "")
        try:
            t_dt = datetime.fromisoformat(t_ts)
        except Exception:
            continue
        delta = abs((t_dt - v_dt).total_seconds())
        if delta < best_delta and delta < 600:
            best_delta = delta
            best_match = t

    if best_match:
        profit = float(best_match.get("profit", 0) or 0)
        vetoed_match_pl += profit
        if profit > 0:
            vetoed_match_wins += 1
        else:
            vetoed_match_losses += 1
    else:
        vetoed_unmatched += 1

matched = vetoed_match_wins + vetoed_match_losses
print(f"  Matched to closed trades: {matched}")
print(f"  Unmatched (no outcome):   {vetoed_unmatched}")
if matched:
    wr = vetoed_match_wins / matched * 100
    print(f"  Wins: {vetoed_match_wins}  Losses: {vetoed_match_losses}  WR: {wr:.1f}%")
    print(f"  Hypothetical P/L of blocked trades: ${vetoed_match_pl:+.2f}")
    if vetoed_match_pl < 0:
        print(f"  >> GPT SAVED ${abs(vetoed_match_pl):.2f} by blocking these trades")
    else:
        print(f"  >> GPT cost ${vetoed_match_pl:.2f} by blocking winners")

# Allowed trades that actually closed — how did they do
print("\n" + "=" * 70)
print("  ALLOWED TRADES — ACTUAL OUTCOMES")
print("=" * 70)

veto_phase_closed = []
for t in closed_trades:
    tags = t.get("tags", {}) or {}
    ds = t.get("decision_source") or tags.get("decision_source", "")
    if ds == "GPT_ASSIST":
        veto_phase_closed.append(t)

if not veto_phase_closed:
    veto_closed_count = 0
    for t in closed_trades:
        t_ts = t.get("timestamp", "")
        try:
            t_dt = datetime.fromisoformat(t_ts)
        except Exception:
            continue
        if shadow:
            try:
                start_dt = datetime.fromisoformat(shadow[0].get("ts", ""))
                if t_dt >= start_dt:
                    veto_phase_closed.append(t)
            except Exception:
                continue

wins = [t for t in veto_phase_closed if float(t.get("profit", 0) or 0) > 0]
losses = [t for t in veto_phase_closed if float(t.get("profit", 0) or 0) <= 0]
total_pl = sum(float(t.get("profit", 0) or 0) for t in veto_phase_closed)
total_win = sum(float(t.get("profit", 0) or 0) for t in wins)
total_loss = sum(float(t.get("profit", 0) or 0) for t in losses)

print(f"  Closed trades during veto phase: {len(veto_phase_closed)}")
print(f"  Wins: {len(wins)}  Losses: {len(losses)}")
if veto_phase_closed:
    wr = len(wins) / len(veto_phase_closed) * 100
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Total P/L: ${total_pl:+.2f}")
    print(f"  Total Wins: ${total_win:+.2f}  Total Losses: ${total_loss:+.2f}")
    if total_loss != 0:
        rr = abs(total_win / total_loss)
        print(f"  R:R Ratio: {rr:.2f}")
    if wins:
        print(f"  Avg Win: ${total_win/len(wins):+.2f}")
    if losses:
        print(f"  Avg Loss: ${total_loss/len(losses):+.2f}")

# By group
print("\n" + "=" * 70)
print("  VETOED BY GROUP")
print("=" * 70)
print(f"  {'GROUP':<12} {'VETOED':>7} {'REASON SAMPLE'}")
print("-" * 70)

veto_groups = {}
for v in vetoed:
    g = v.get("group", "unknown")
    if g not in veto_groups:
        veto_groups[g] = {"count": 0, "reasons": []}
    veto_groups[g]["count"] += 1
    reason = v.get("gpt_reasoning", "")[:60]
    if reason and len(veto_groups[g]["reasons"]) < 2:
        veto_groups[g]["reasons"].append(reason)

for g in sorted(veto_groups, key=lambda x: veto_groups[x]["count"], reverse=True):
    d = veto_groups[g]
    reason_str = d["reasons"][0] if d["reasons"] else ""
    print(f"  {g:<12} {d['count']:>7}  {reason_str}")

# By symbol
print("\n" + "=" * 70)
print("  VETOED BY SYMBOL (top 15)")
print("=" * 70)
print(f"  {'SYMBOL':<14} {'VETOED':>7} {'GPT_CONF':>9}")
print("-" * 70)

veto_syms = {}
for v in vetoed:
    s = v.get("symbol", "unknown")
    if s not in veto_syms:
        veto_syms[s] = {"count": 0, "confs": []}
    veto_syms[s]["count"] += 1
    c = v.get("gpt_confidence", 0)
    if c:
        veto_syms[s]["confs"].append(c)

for s in sorted(veto_syms, key=lambda x: veto_syms[x]["count"], reverse=True)[:15]:
    d = veto_syms[s]
    avg_conf = sum(d["confs"]) / len(d["confs"]) if d["confs"] else 0
    print(f"  {s:<14} {d['count']:>7}   avg {avg_conf:.0f}%")

print("\n" + "=" * 70)
