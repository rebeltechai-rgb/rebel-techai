"""
GPT Veto Phase Review — what has GPT done since veto went live?
"""
import json
import os
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
TRADES_PATH = os.path.join(LOG_DIR, "trades.jsonl")
VETO_LOG_PATH = os.path.join(LOG_DIR, "veto_log.jsonl")

all_trades = []
with open(TRADES_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            all_trades.append(json.loads(line))
        except Exception:
            continue

print(f"[DEBUG] Total trade records loaded: {len(all_trades)}")

# Find GPT_ASSIST trades (veto phase — GPT allowed these through)
veto_opened = []
veto_closed = []

for t in all_trades:
    tags = t.get("tags", {}) or {}
    ds = tags.get("decision_source", "") or t.get("decision_source", "")

    if ds != "GPT_ASSIST":
        continue

    direction = (t.get("direction") or "").upper()
    status = (t.get("status") or "").upper()

    if status == "CLOSED" or direction == "CLOSE":
        veto_closed.append(t)
    elif direction in ("BUY", "SELL"):
        veto_opened.append(t)

# Also find closed trades after veto start by timestamp
# (catches trades opened during veto that closed without GPT_ASSIST tag)
veto_ts_cutoff = "2026-02-26T12:00:00"  # approximate veto start
all_closed_after_veto = []
for t in all_trades:
    status = (t.get("status") or "").upper()
    direction = (t.get("direction") or "").upper()
    if status != "CLOSED" and direction != "CLOSE":
        continue
    ts = t.get("timestamp", "")
    if ts >= veto_ts_cutoff:
        all_closed_after_veto.append(t)

print("=" * 70)
print("  GPT VETO PHASE REVIEW")
print("=" * 70)
print(f"  Trades opened with GPT_ASSIST tag: {len(veto_opened)}")
print(f"  Trades closed with GPT_ASSIST tag: {len(veto_closed)}")
print(f"  All closed trades since veto start: {len(all_closed_after_veto)}")

# Use the larger set for analysis
closed = all_closed_after_veto if len(all_closed_after_veto) > len(veto_closed) else veto_closed

if closed:
    wins = [t for t in closed if float(t.get("profit", 0) or 0) > 0]
    losses = [t for t in closed if float(t.get("profit", 0) or 0) <= 0]
    total_pl = sum(float(t.get("profit", 0) or 0) for t in closed)
    total_win = sum(float(t.get("profit", 0) or 0) for t in wins)
    total_loss = sum(float(t.get("profit", 0) or 0) for t in losses)

    print(f"\n{'='*70}")
    print(f"  VETO PHASE — CLOSED TRADE OUTCOMES")
    print(f"{'='*70}")
    print(f"  Total closed: {len(closed)}")
    print(f"  Wins: {len(wins)}  Losses: {len(losses)}")
    if closed:
        wr = len(wins) / len(closed) * 100
        print(f"  Win Rate: {wr:.1f}%")
    print(f"  Total P/L: ${total_pl:+.2f}")
    print(f"  Wins Total: ${total_win:+.2f}  Losses Total: ${total_loss:+.2f}")
    if total_loss != 0:
        rr = abs(total_win / total_loss)
        print(f"  R:R Ratio: {rr:.2f}")
    if wins:
        print(f"  Avg Win: ${total_win/len(wins):+.2f}")
    if losses:
        print(f"  Avg Loss: ${total_loss/len(losses):+.2f}")

    # By group
    print(f"\n{'='*70}")
    print(f"  BY GROUP")
    print(f"{'='*70}")
    print(f"  {'GROUP':<12} {'TRADES':>7} {'WINS':>6} {'LOSSES':>7} {'WR%':>6} {'P/L':>10}")
    print(f"  {'-'*55}")

    groups = {}
    for t in closed:
        tags = t.get("tags", {}) or {}
        g = (t.get("group") or tags.get("group") or "unknown").lower()
        if g not in groups:
            groups[g] = {"trades": 0, "wins": 0, "losses": 0, "pl": 0.0}
        groups[g]["trades"] += 1
        p = float(t.get("profit", 0) or 0)
        groups[g]["pl"] += p
        if p > 0:
            groups[g]["wins"] += 1
        else:
            groups[g]["losses"] += 1

    for g in sorted(groups, key=lambda x: groups[x]["pl"], reverse=True):
        d = groups[g]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(f"  {g:<12} {d['trades']:>7} {d['wins']:>6} {d['losses']:>7} {wr:>5.1f}% {d['pl']:>+10.2f}")

    # By symbol (top 10 best and worst)
    print(f"\n{'='*70}")
    print(f"  TOP 10 SYMBOLS (best P/L)")
    print(f"{'='*70}")
    print(f"  {'SYMBOL':<14} {'TRADES':>7} {'WINS':>6} {'LOSSES':>7} {'WR%':>6} {'P/L':>10}")
    print(f"  {'-'*55}")

    syms = {}
    for t in closed:
        s = (t.get("symbol") or "unknown").upper()
        if s not in syms:
            syms[s] = {"trades": 0, "wins": 0, "losses": 0, "pl": 0.0}
        syms[s]["trades"] += 1
        p = float(t.get("profit", 0) or 0)
        syms[s]["pl"] += p
        if p > 0:
            syms[s]["wins"] += 1
        else:
            syms[s]["losses"] += 1

    sorted_syms = sorted(syms, key=lambda x: syms[x]["pl"], reverse=True)
    for s in sorted_syms[:10]:
        d = syms[s]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(f"  {s:<14} {d['trades']:>7} {d['wins']:>6} {d['losses']:>7} {wr:>5.1f}% {d['pl']:>+10.2f}")

    print(f"\n{'='*70}")
    print(f"  BOTTOM 10 SYMBOLS (worst P/L)")
    print(f"{'='*70}")
    print(f"  {'SYMBOL':<14} {'TRADES':>7} {'WINS':>6} {'LOSSES':>7} {'WR%':>6} {'P/L':>10}")
    print(f"  {'-'*55}")
    for s in sorted_syms[-10:]:
        d = syms[s]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(f"  {s:<14} {d['trades']:>7} {d['wins']:>6} {d['losses']:>7} {wr:>5.1f}% {d['pl']:>+10.2f}")
else:
    print("\n  No closed trades found during veto phase.")

# Veto block log (new — starts recording after next restart)
if os.path.exists(VETO_LOG_PATH):
    veto_blocks = []
    with open(VETO_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    veto_blocks.append(json.loads(line))
                except Exception:
                    continue

    if veto_blocks:
        print(f"\n{'='*70}")
        print(f"  GPT VETO BLOCKS (trades GPT killed)")
        print(f"{'='*70}")
        print(f"  Total blocked: {len(veto_blocks)}")

        vg = {}
        for v in veto_blocks:
            g = v.get("group", "unknown")
            vg[g] = vg.get(g, 0) + 1
        print(f"\n  By group:")
        for g in sorted(vg, key=lambda x: vg[x], reverse=True):
            print(f"    {g:<12} {vg[g]:>5} blocked")

        vs = {}
        for v in veto_blocks:
            s = v.get("symbol", "unknown")
            vs[s] = vs.get(s, 0) + 1
        print(f"\n  By symbol (top 10):")
        for s in sorted(vs, key=lambda x: vs[x], reverse=True)[:10]:
            print(f"    {s:<14} {vs[s]:>5} blocked")

        print(f"\n  Recent blocks:")
        for v in veto_blocks[-5:]:
            print(f"    {v.get('ts','?')[:19]} | {v.get('symbol','?'):<12} | {v.get('session','?'):<8} | {v.get('gpt_reasoning','')[:50]}")
else:
    print(f"\n  [NOTE] Veto block log not yet active — will start after Trader restart.")

print(f"\n{'='*70}")
