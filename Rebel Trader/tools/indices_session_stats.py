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
        if 7 <= h < 13:
            return "LONDON"
        elif 13 <= h < 21:
            return "NEW_YORK"
        elif 0 <= h < 7 or h >= 21:
            return "TOKYO"
        else:
            return "OTHER"
    except:
        return "unknown"

# Get all index symbols from trades
index_group = ["indices"]
index_trades = [t for t in closed if t.get("group", "").lower() in index_group]

# Unique symbols
symbols = sorted(set(t["symbol"] for t in index_trades))

print("=" * 75)
print("  INDICES - ALL SYMBOLS BY SESSION")
print("=" * 75)

# Overall indices by session
for sess in ["LONDON", "NEW_YORK", "TOKYO"]:
    st = [t for t in index_trades if get_session(t.get("close_time")) == sess]
    if not st:
        continue
    wins = sum(1 for t in st if t["outcome"] == "WIN")
    losses = sum(1 for t in st if t["outcome"] == "LOSS")
    total = len(st)
    pnl = sum(float(t.get("profit", 0)) for t in st)
    wr = wins / total * 100 if total else 0
    avg_w = sum(float(t.get("profit", 0)) for t in st if t["outcome"] == "WIN") / wins if wins else 0
    avg_l = sum(float(t.get("profit", 0)) for t in st if t["outcome"] == "LOSS") / losses if losses else 0
    print(f"  {sess:<12} Trades:{total:>3}  W:{wins:>3}  L:{losses:>3}  WR:{wr:>5.1f}%  P/L:{pnl:>+9.2f}  AvgW:{avg_w:>+8.2f}  AvgL:{avg_l:>+8.2f}")

lon_ny = [t for t in index_trades if get_session(t.get("close_time")) in ("LONDON", "NEW_YORK")]
if lon_ny:
    wins = sum(1 for t in lon_ny if t["outcome"] == "WIN")
    losses = sum(1 for t in lon_ny if t["outcome"] == "LOSS")
    total = len(lon_ny)
    pnl = sum(float(t.get("profit", 0)) for t in lon_ny)
    wr = wins / total * 100 if total else 0
    print(f"  {'LON+NY':<12} Trades:{total:>3}  W:{wins:>3}  L:{losses:>3}  WR:{wr:>5.1f}%  P/L:{pnl:>+9.2f}")

total_all = len(index_trades)
pnl_all = sum(float(t.get("profit", 0)) for t in index_trades)
wins_all = sum(1 for t in index_trades if t["outcome"] == "WIN")
losses_all = sum(1 for t in index_trades if t["outcome"] == "LOSS")
wr_all = wins_all / total_all * 100 if total_all else 0
print(f"  {'TOTAL':<12} Trades:{total_all:>3}  W:{wins_all:>3}  L:{losses_all:>3}  WR:{wr_all:>5.1f}%  P/L:{pnl_all:>+9.2f}")

# Per-symbol breakdown
for sym in symbols:
    sym_trades = [t for t in index_trades if t["symbol"] == sym]
    print()
    print(f"  --- {sym} ---")
    for sess in ["LONDON", "NEW_YORK", "TOKYO"]:
        st = [t for t in sym_trades if get_session(t.get("close_time")) == sess]
        if not st:
            continue
        wins = sum(1 for t in st if t["outcome"] == "WIN")
        losses = sum(1 for t in st if t["outcome"] == "LOSS")
        total = len(st)
        pnl = sum(float(t.get("profit", 0)) for t in st)
        wr = wins / total * 100 if total else 0
        avg_w = sum(float(t.get("profit", 0)) for t in st if t["outcome"] == "WIN") / wins if wins else 0
        avg_l = sum(float(t.get("profit", 0)) for t in st if t["outcome"] == "LOSS") / losses if losses else 0
        print(f"    {sess:<12} Trades:{total:>3}  W:{wins:>3}  L:{losses:>3}  WR:{wr:>5.1f}%  P/L:{pnl:>+9.2f}  AvgW:{avg_w:>+8.2f}  AvgL:{avg_l:>+8.2f}")
    total_s = len(sym_trades)
    pnl_s = sum(float(t.get("profit", 0)) for t in sym_trades)
    wins_s = sum(1 for t in sym_trades if t["outcome"] == "WIN")
    wr_s = wins_s / total_s * 100 if total_s else 0
    print(f"    {'TOTAL':<12} Trades:{total_s:>3}  W:{wins_s:>3}  L:{total_s - wins_s:>3}  WR:{wr_s:>5.1f}%  P/L:{pnl_s:>+9.2f}")
