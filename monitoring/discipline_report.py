from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional


ROOT = r"C:\Rebel Technologies"

MASTER_LOGS = os.path.join(ROOT, "Rebel Master", "logs")
TRADER_LOGS = os.path.join(ROOT, "Rebel Trader", "logs")


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _within_lookback(ts: Optional[datetime], lookback: timedelta) -> bool:
    if ts is None:
        return False
    return ts >= datetime.now(timezone.utc) - lookback


def _tail_lines(path: str, max_lines: int = 2000) -> List[str]:
    if not os.path.exists(path):
        return []
    size = os.path.getsize(path)
    if size == 0:
        return []
    with open(path, "rb") as f:
        block = 4096
        data = b""
        while size > 0 and data.count(b"\n") <= max_lines:
            read_size = block if size >= block else size
            size -= read_size
            f.seek(size)
            data = f.read(read_size) + data
        lines = data.splitlines()
    return [line.decode(errors="ignore") for line in lines[-max_lines:]]


def master_trade_summary(lookback: timedelta) -> None:
    trades_path = os.path.join(MASTER_LOGS, "trades.csv")
    if not os.path.exists(trades_path):
        print("[MASTER] trades.csv not found")
        return

    total = 0
    failures = 0
    last_ts = None
    failure_reasons = Counter()
    volume_blocks = 0

    with open(trades_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_iso(row.get("timestamp", ""))
            if not _within_lookback(ts, lookback):
                continue
            total += 1
            last_ts = max(last_ts, ts) if last_ts else ts
            result = (row.get("result") or "").strip()
            if result:
                failures += 1
                failure_reasons[result] += 1
                if "volume_below_min" in result or "invalid_lot" in result:
                    volume_blocks += 1

    print(f"[MASTER] Trades (last {lookback.days}d): {total} | Failures: {failures}")
    if last_ts:
        print(f"[MASTER] Last trade: {last_ts.isoformat()}")
    if volume_blocks:
        print(f"[MASTER] Volume blocks: {volume_blocks}")
    if failure_reasons:
        top = ", ".join(f"{k}={v}" for k, v in failure_reasons.most_common(5))
        print(f"[MASTER] Failure reasons: {top}")


def master_filter_summary(lookback: timedelta) -> None:
    path = os.path.join(MASTER_LOGS, "filter_rejections.txt")
    if not os.path.exists(path):
        print("[MASTER] filter_rejections.txt not found")
        return
    reasons = Counter()
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            ts = _parse_iso(parts[0]) if parts else None
            if not _within_lookback(ts, lookback):
                continue
            total += 1
            for p in parts:
                if p.startswith("Reason="):
                    reasons[p.replace("Reason=", "")] += 1
                    break

    print(f"[MASTER] Filter rejections (last {lookback.days}d): {total}")
    if reasons:
        top = ", ".join(f"{k}={v}" for k, v in reasons.most_common(5))
        print(f"[MASTER] Top reasons: {top}")


def trader_trade_summary(lookback: timedelta) -> None:
    path = os.path.join(TRADER_LOGS, "trades.jsonl")
    if not os.path.exists(path):
        print("[TRADER] trades.jsonl not found")
        return

    total = 0
    dry = 0
    last_ts = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = _parse_iso(rec.get("timestamp", ""))
            if not _within_lookback(ts, lookback):
                continue
            total += 1
            if rec.get("dry_run"):
                dry += 1
            last_ts = max(last_ts, ts) if last_ts else ts

    print(f"[TRADER] Trades (last {lookback.days}d): {total} | Dry-run: {dry}")
    if last_ts:
        print(f"[TRADER] Last trade: {last_ts.isoformat()}")


def trader_log_anomalies() -> None:
    path = os.path.join(TRADER_LOGS, "rebel_trader.log")
    lines = _tail_lines(path, max_lines=2000)
    if not lines:
        print("[TRADER] rebel_trader.log not found or empty")
        return

    blocked = sum(1 for l in lines if "[BLOCKED]" in l)
    failed = sum(1 for l in lines if "[FAILED]" in l or "Error" in l or "ERROR" in l)
    print(f"[TRADER] Recent log flags: BLOCKED={blocked} | FAILED/ERROR={failed}")


def main() -> None:
    lookback = timedelta(days=1)
    print("=== REBEL DISCIPLINE REPORT (last 24h) ===")
    master_trade_summary(lookback)
    master_filter_summary(lookback)
    trader_trade_summary(lookback)
    trader_log_anomalies()


if __name__ == "__main__":
    main()
