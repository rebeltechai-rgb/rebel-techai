import json
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "gpt_shadow_log.jsonl")
TRADES_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "trades.jsonl")

records = []
with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

trades = []
with open(TRADES_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

closed = {t["symbol"] + "_" + t.get("open_time", ""): t for t in trades if t.get("status") == "CLOSED" and t.get("outcome")}

total = len(records)
agree = sum(1 for r in records if r.get("agree"))
disagree = total - agree

gpt_would_hold = sum(1 for r in records if r.get("gpt_direction", "").upper() == "HOLD")
gpt_would_trade = total - gpt_would_hold

rules_hold = sum(1 for r in records if r.get("rules_direction", "").upper() == "HOLD")
rules_trade = total - rules_hold

print("=" * 65)
print("  GPT SHADOW LOG STATS")
print(f"  Total shadow decisions logged: {total}")
print("=" * 65)

print(f"\n  Agreement rate: {agree}/{total} ({agree/total*100:.1f}%)" if total else "")
print(f"  Disagreements:  {disagree}/{total} ({disagree/total*100:.1f}%)" if total else "")

print(f"\n  GPT said TRADE: {gpt_would_trade}  |  GPT said HOLD: {gpt_would_hold}")
print(f"  Rules said TRADE: {rules_trade}  |  Rules said HOLD: {rules_hold}")

# When they disagreed, what happened
print("\n" + "=" * 65)
print("  DISAGREEMENT ANALYSIS")
print("=" * 65)

gpt_block_rules_trade = 0
gpt_trade_rules_hold = 0
for r in records:
    if not r.get("agree"):
        gpt_dir = r.get("gpt_direction", "HOLD").upper()
        rules_dir = r.get("rules_direction", "HOLD").upper()
        if gpt_dir == "HOLD" and rules_dir != "HOLD":
            gpt_block_rules_trade += 1
        elif gpt_dir != "HOLD" and rules_dir == "HOLD":
            gpt_trade_rules_hold += 1

print(f"  GPT would BLOCK, Rules traded:    {gpt_block_rules_trade}")
print(f"  GPT would TRADE, Rules held:      {gpt_trade_rules_hold}")
print(f"  Direction mismatch (both trade):   {disagree - gpt_block_rules_trade - gpt_trade_rules_hold}")

# GPT confidence stats
confs = [r.get("gpt_confidence", 0) for r in records if r.get("gpt_confidence")]
if confs:
    print(f"\n  GPT avg confidence: {sum(confs)/len(confs):.1f}%")
    print(f"  GPT min confidence: {min(confs)}%")
    print(f"  GPT max confidence: {max(confs)}%")

# By group
print("\n" + "=" * 65)
print("  BY GROUP")
print("=" * 65)
print(f"  {'GROUP':<12} {'TOTAL':>6} {'AGREE':>6} {'AGREE%':>7} {'GPT_HOLD':>9} {'GPT_TRADE':>10}")
print("-" * 65)

groups = {}
for r in records:
    g = r.get("group", "unknown")
    if g not in groups:
        groups[g] = {"total": 0, "agree": 0, "gpt_hold": 0, "gpt_trade": 0}
    groups[g]["total"] += 1
    if r.get("agree"):
        groups[g]["agree"] += 1
    if r.get("gpt_direction", "HOLD").upper() == "HOLD":
        groups[g]["gpt_hold"] += 1
    else:
        groups[g]["gpt_trade"] += 1

for g in sorted(groups, key=lambda x: groups[x]["total"], reverse=True):
    d = groups[g]
    ar = d["agree"] / d["total"] * 100 if d["total"] else 0
    print(f"  {g:<12} {d['total']:>6} {d['agree']:>6} {ar:>6.1f}% {d['gpt_hold']:>9} {d['gpt_trade']:>10}")

# Veto preview: if GPT had veto power, how many trades would it have blocked
print("\n" + "=" * 65)
print("  VETO PREVIEW")
print("  (If GPT had veto power from the start)")
print("=" * 65)

traded = [r for r in records if r.get("trade_placed")]
would_veto = [r for r in traded if r.get("gpt_direction", "").upper() == "HOLD"]
would_allow = [r for r in traded if r.get("gpt_direction", "").upper() != "HOLD"]

print(f"  Trades placed by rules:     {len(traded)}")
print(f"  GPT would have VETOED:      {len(would_veto)}")
print(f"  GPT would have ALLOWED:     {len(would_allow)}")
if traded:
    print(f"  Veto rate:                  {len(would_veto)/len(traded)*100:.1f}%")
