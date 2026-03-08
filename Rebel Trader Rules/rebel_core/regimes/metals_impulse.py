from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import MetaTrader5 as mt5

SESSION_ORDER = ["TOKYO", "LONDON", "NEW_YORK"]


def previous_session_name(current_session: str) -> str:
    if current_session not in SESSION_ORDER:
        return "NEW_YORK"
    idx = SESSION_ORDER.index(current_session)
    return SESSION_ORDER[idx - 1] if idx > 0 else SESSION_ORDER[-1]


def get_session_window(session_name: str, now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    now = now or datetime.now(timezone.utc)
    session = session_name.upper()

    if session == "LONDON":
        start = now.replace(hour=7, minute=0, second=0, microsecond=0)
        end = now.replace(hour=13, minute=0, second=0, microsecond=0)
        if now < start:
            start -= timedelta(days=1)
            end -= timedelta(days=1)
        return start, end

    if session == "NEW_YORK":
        start = now.replace(hour=13, minute=0, second=0, microsecond=0)
        end = now.replace(hour=21, minute=0, second=0, microsecond=0)
        if now < start:
            start -= timedelta(days=1)
            end -= timedelta(days=1)
        return start, end

    # TOKYO (default): 21:00 - 07:00 (spans midnight)
    end = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now < end:
        start = (end - timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
    else:
        start = end.replace(hour=21, minute=0, second=0, microsecond=0)
        if start > now:
            start -= timedelta(days=1)
    return start, end


def detect_metals_impulse(session_stats: Dict[str, Any], last_session_stats: Dict[str, Any]) -> bool:
    if (
        last_session_stats.get("metals", {}).get("blue_trades", 0) >= 2
        and last_session_stats.get("metals", {}).get("avg_win", 0) > session_stats.get("global", {}).get("avg_win", 0)
        and session_stats.get("metals", {}).get("first_trade_direction")
            == last_session_stats.get("metals", {}).get("dominant_direction")
    ):
        return True
    return False


def build_session_stats(
    deals: Iterable[Any],
    groups: Dict[str, List[str]],
) -> Dict[str, Any]:
    stats = {
        "global": {"avg_win": 0.0},
        "metals": {
            "blue_trades": 0,
            "avg_win": 0.0,
            "first_trade_direction": None,
            "dominant_direction": None,
            "losses": 0,
        },
        "crypto": {"losses": 0},
    }

    symbol_to_group = {}
    for group_name, symbols in (groups or {}).items():
        if isinstance(symbols, list):
            for symbol in symbols:
                symbol_to_group[symbol] = group_name.lower()

    wins = []
    metals_wins = []
    metals_losses = 0
    crypto_losses = 0

    direction_counts = {"BUY": 0, "SELL": 0}
    first_direction = None

    for deal in deals or []:
        symbol = getattr(deal, "symbol", "")
        group = symbol_to_group.get(symbol, "")
        entry = getattr(deal, "entry", None)
        profit = float(getattr(deal, "profit", 0.0) or 0.0)

        # Determine direction on entry deals
        if entry == mt5.DEAL_ENTRY_IN:
            deal_type = getattr(deal, "type", None)
            direction = "BUY" if deal_type in (mt5.ORDER_TYPE_BUY, mt5.DEAL_TYPE_BUY) else "SELL"
            direction_counts[direction] += 1
            if first_direction is None:
                first_direction = direction

        # Use closing deals to evaluate wins/losses
        if entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY):
            if profit > 0:
                wins.append(profit)
                if group == "metals":
                    metals_wins.append(profit)
            else:
                if group == "metals":
                    metals_losses += 1
                if group == "crypto":
                    crypto_losses += 1

    if wins:
        stats["global"]["avg_win"] = sum(wins) / len(wins)
    if metals_wins:
        stats["metals"]["avg_win"] = sum(metals_wins) / len(metals_wins)
        stats["metals"]["blue_trades"] = len(metals_wins)

    stats["metals"]["losses"] = metals_losses
    stats["crypto"]["losses"] = crypto_losses
    stats["metals"]["first_trade_direction"] = first_direction
    stats["metals"]["dominant_direction"] = "BUY" if direction_counts["BUY"] >= direction_counts["SELL"] else "SELL"

    return stats

