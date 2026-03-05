"""
GPT Veto Phase Review
Analyzes trades that went through during the veto phase (decision_source=GPT_ASSIST)
and reads the veto_log.jsonl for blocked trades (if available).
"""
import json
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
TRADES_PATH = os.path.join(LOG_DIR, "trades.jsonl")
VETO_LOG_PATH = os.path.join(LOG_DIR, "veto_log.jsonl")
SHADOW_PATH = os.path.join(LOG_DIR, "gpt_shadow_log.jsonl")

# --- Find the timestamp when veto started (governed_count ~ 1250) ---
veto_start_ts = None
if os.path.exists(SHADOW_PATH):
    last_shadow_ts = None
    with open(SHADOW_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                gc = int(rec.get("governed_count", 0) or 0)
                if gc >= 1245 and veto_start_ts is None:
                    veto_start_ts = rec.get("ts")
                last_shadow_ts = rec.get("ts")
            except Exception:
                continue
    if not veto_start_ts and last_shadow_ts:
        veto_start_ts = last_shadow_ts

# --- Load all trades ---
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

# --- Split into veto-phase trades ---
veto_allowed = []
veto_closed = []
all_closed_veto = []

from datetime import datetime

for t in all_trades:
    tags = t.get("tags", {}) or {}
    ds = t.get("decision_source") or tags.get("decision_source", "")

    is_veto_phase = ds == "GPT_ASSIST"

    if not is_veto_phase and veto_start_ts:
        t_ts = t.get("timestamp", "")
        try:
            t_dt = datetime.fromisoformat(t_ts)
            v_dt = datetime.fromisoformat(veto_start_ts)
            if t_dt >= v_dt:
                is_veto_phase = True
        except Exception:
            pass

    if not is_veto_phase:
        continue

    direction = t.get("direction", "")
    status = t.get("status", "")

    if status == "CLOSED" or direction == "CLOSE":
        all_closed_veto.append(t)
    elif direction in ("BUY", "SELL"):
        veto_allowed.append(t)

# --- Load veto block log (new feature) ---
veto_blocks = []
if os.path.exists(VETO_LOG_PATH):
    with open(VETO_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                veto_blocks.append(json.loads(line))
            except Exception:
                continue

# --- Print report ---
print("=" * 70)
print("  GPT VETO PHASE REVIEW")
print("=" * 70)
if veto_start_ts:
    print(f"  Veto active since: {veto_start_ts[:19]}")
print(f"  Trades opened during veto phase: {len(veto_allowed)}")
print(f"  Trades closed during veto phase: {len(all_closed_veto)}")
print(f"  Vetoed (blocked) trades logged:  {len(veto_blocks)}")

# --- Closed trade stats ---
if all_closed_veto:
    wins = [t for t in all_closed_veto if float(t.get("profit", 0) or 0) > 0]
    losses = [t for t in all_closed_veto if float(t.get("profit", 0) or 0) <= 0]
    total_pl = sum(float(t.get("profit", 0) or 0) for t in all_closed_veto)
    total_win = sum(float(t.get("profit", 0) or 0) for t in wins)
    total_loss = sum(float(t.get("profit", 0) or 0) for t in losses)

    print(f"\n  {'='*60}")
    print(f"  ALLOWED TRADES — CLOSED OUTCOMES")
    print(f"  {'='*60}")
    print(f"  Wins: {len(wins)}  Losses: {len(losses)}")
    wr = len(wins) / len(all_closed_veto) * 100
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
    print(f"\n  {'='*60}")
    print(f"  BY GROUP")
    print(f"  {'='*60}")
    print(f"  {'GROUP':<12} {'TRADES':>7} {'WINS':>6} {'LOSSES':>7} {'WR%':>6} {'P/L':>10}")
    print(f"  {'-'*55}")

    groups = {}
    for t in all_closed_veto:
        g = (t.get("group") or "unknown").lower()
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

# --- Veto blocks ---
if veto_blocks:
    print(f"\n  {'='*60}")
    print(f"  BLOCKED TRADES (GPT VETOES)")
    print(f"  {'='*60}")
    print(f"  Total blocked: {len(veto_blocks)}")

    vg = {}
    for v in veto_blocks:
        g = v.get("group", "unknown")
        if g not in vg:
            vg[g] = 0
        vg[g] += 1

    print(f"\n  By group:")
    for g in sorted(vg, key=lambda x: vg[x], reverse=True):
        print(f"    {g:<12} {vg[g]:>5} blocked")

    vs = {}
    for v in veto_blocks:
        s = v.get("symbol", "unknown")
        if s not in vs:
            vs[s] = 0
        vs[s] += 1

    print(f"\n  By symbol (top 10):")
    for s in sorted(vs, key=lambda x: vs[x], reverse=True)[:10]:
        print(f"    {s:<14} {vs[s]:>5} blocked")
else:
    print(f"\n  [NOTE] No veto block log found yet.")
    print(f"  Veto blocks will be logged going forward (veto_log.jsonl).")

print(f"\n{'='*70}")
