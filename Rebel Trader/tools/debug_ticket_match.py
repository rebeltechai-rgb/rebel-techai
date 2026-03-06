"""Debug ticket matching between entries and closes in trades.jsonl"""
import json
from collections import Counter

TRADES_FILE = r"C:\Rebel Technologies\Rebel Trader\logs\trades.jsonl"

entries_by_ticket = {}
closes_by_ticket = {}
gpt_tickets = set()
all_lines = []

with open(TRADES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except Exception:
            continue
        all_lines.append(t)

        ticket = t.get("ticket")
        status = t.get("status", "")
        direction = t.get("direction", "")
        ds = t.get("decision_source", "")

        if ds == "GPT_ASSIST":
            gpt_tickets.add(ticket)

        if status == "CLOSED":
            closes_by_ticket[ticket] = t
        elif direction in ("BUY", "SELL") and ds:
            entries_by_ticket[ticket] = t

gpt_in_closes = gpt_tickets & set(closes_by_ticket.keys())
overlap = set(entries_by_ticket.keys()) & set(closes_by_ticket.keys())

print(f"GPT entry tickets:            {len(gpt_tickets)}")
print(f"GPT tickets found in closes:  {len(gpt_in_closes)}")
print(f"Total entry tickets:          {len(entries_by_ticket)}")
print(f"Total close tickets:          {len(closes_by_ticket)}")
print(f"Entry-Close ticket overlap:   {len(overlap)}")
print(f"Total lines:                  {len(all_lines)}")

# Entries without decision_source (not tagged)
no_ds = [t for t in all_lines if t.get("direction") in ("BUY", "SELL") and not t.get("decision_source")]
print(f"\nEntries without decision_source: {len(no_ds)}")

# All direction values
dirs = Counter(t.get("direction", "") for t in all_lines)
print(f"\nDirection distribution:")
for d, cnt in dirs.most_common():
    print(f"  {d}: {cnt}")

# All status values
statuses = Counter(t.get("status", "(none)") for t in all_lines)
print(f"\nStatus distribution:")
for s, cnt in statuses.most_common():
    print(f"  {s}: {cnt}")

# Sample a GPT entry
if gpt_tickets:
    tk = sorted(gpt_tickets)[0]
    e = entries_by_ticket.get(tk, {})
    print(f"\nSample GPT entry ticket {tk}:")
    print(f"  symbol={e.get('symbol')} direction={e.get('direction')} ts={e.get('timestamp', '')[:16]}")

# Sample closes
print(f"\nLast 5 closes:")
close_list = list(closes_by_ticket.items())[-5:]
for tk, c in close_list:
    print(f"  ticket={tk} sym={c.get('symbol')} outcome={c.get('outcome')} pnl={c.get('profit')}")
