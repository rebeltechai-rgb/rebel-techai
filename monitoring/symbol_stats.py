import csv
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


LABELS_PATH = r"C:\Rebel Technologies\Rebel Master\ML\labels.csv"


def _parse_time(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    ts = ts.strip()
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def load_symbol_stats(path: str, days: Optional[int]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    cutoff = None
    if days is not None and days > 0:
        cutoff = datetime.now() - timedelta(days=days)

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_time(row.get("timestamp", ""))
            if cutoff and ts and ts < cutoff:
                continue
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                continue

            pnl = _safe_float(row.get("pnl"))
            rr = _safe_float(row.get("rr"))
            label = row.get("label")

            record = stats.setdefault(
                symbol,
                {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "total_rr": 0.0,
                    "rr_count": 0,
                    "last_trade": ts,
                },
            )

            record["trades"] += 1
            if str(label).strip() == "1":
                record["wins"] += 1
            elif str(label).strip() == "0":
                record["losses"] += 1
            if pnl is not None:
                record["total_pnl"] += pnl
            if rr is not None:
                record["total_rr"] += rr
                record["rr_count"] += 1
            if ts and (record["last_trade"] is None or ts > record["last_trade"]):
                record["last_trade"] = ts

    for record in stats.values():
        trades = record["trades"]
        record["win_rate"] = (record["wins"] / trades) * 100 if trades else 0.0
        record["avg_pnl"] = record["total_pnl"] / trades if trades else 0.0
        record["avg_rr"] = record["total_rr"] / record["rr_count"] if record["rr_count"] else 0.0

    return stats


def render_table(rows: List[Dict[str, Any]]) -> None:
    print("=" * 110)
    print("REBEL MASTER - SYMBOL STATS (from labels.csv)")
    print("=" * 110)
    print(f"{'Symbol':<12} {'Trades':>6} {'Wins':>6} {'Losses':>7} {'Win%':>7} {'TotalPnL':>12} {'AvgPnL':>10} {'AvgRR':>8} {'Last Trade':>20}")
    print("-" * 110)
    for r in rows:
        last_trade = r["last_trade"].strftime("%Y-%m-%d %H:%M") if r["last_trade"] else "n/a"
        print(
            f"{r['symbol']:<12} {r['trades']:>6} {r['wins']:>6} {r['losses']:>7} "
            f"{r['win_rate']:>6.1f}% {r['total_pnl']:>12.4f} {r['avg_pnl']:>10.4f} {r['avg_rr']:>8.2f} {last_trade:>20}"
        )
    print("=" * 110)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-symbol stats from Master labels.csv")
    parser.add_argument("--days", type=int, default=None, help="Only include last N days")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum trades per symbol")
    parser.add_argument(
        "--sort",
        choices=["trades", "win_rate", "total_pnl", "avg_pnl", "avg_rr"],
        default="trades",
        help="Sort field",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max symbols to display")
    parser.add_argument("--path", type=str, default=LABELS_PATH, help="Path to labels.csv")
    args = parser.parse_args()

    stats = load_symbol_stats(args.path, args.days)
    rows = []
    for symbol, data in stats.items():
        if data["trades"] < args.min_trades:
            continue
        rows.append({"symbol": symbol, **data})

    rows.sort(key=lambda r: r.get(args.sort) or 0, reverse=True)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        print("No symbol stats available for the selected filters.")
        return

    render_table(rows)


if __name__ == "__main__":
    main()
