from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import MetaTrader5 as mt5


REGIME_HISTORY_PATH = r"C:\Rebel Technologies\Rebel Master\logs\regime_history.json"
REGIME_STATE_PATH = r"C:\Rebel Technologies\Rebel Master\logs\regime_guard_state.json"
TRADE_LIMITER_STATE_PATH = r"C:\Rebel Technologies\Rebel Master\logs\trade_limiter_state.json"

MAGIC_NUMBER = 20251202


def validate_metals_impulse(current_stats: Dict[str, Any]) -> bool:
    if (
        current_stats["metals"]["consecutive_losses"] >= 2
        or current_stats["metals"]["first_red_after_impulse"]
        or current_stats["session_boundary"]
    ):
        return False
    return True


def _get_session_name(now: datetime) -> str:
    hour = now.hour
    weekday = now.weekday()
    if weekday >= 5:
        return "WEEKEND"
    if 12 <= hour < 16:
        return "OVERLAP_LONDON_NY"
    if 7 <= hour < 12:
        return "LONDON"
    if 16 <= hour < 21:
        return "NEW_YORK"
    return "TOKYO"


def _get_session_window(session_name: str, now: datetime) -> Tuple[datetime, datetime]:
    session = session_name.upper()

    if session == "LONDON":
        start = now.replace(hour=7, minute=0, second=0, microsecond=0)
        end = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if now < start:
            start -= timedelta(days=1)
            end -= timedelta(days=1)
        return start, end

    if session == "OVERLAP_LONDON_NY":
        start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now < start:
            start -= timedelta(days=1)
            end -= timedelta(days=1)
        return start, end

    if session == "NEW_YORK":
        start = now.replace(hour=16, minute=0, second=0, microsecond=0)
        end = now.replace(hour=21, minute=0, second=0, microsecond=0)
        if now < start:
            start -= timedelta(days=1)
            end -= timedelta(days=1)
        return start, end

    end = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now < end:
        start = (end - timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
    else:
        start = end.replace(hour=21, minute=0, second=0, microsecond=0)
        if start > now:
            start -= timedelta(days=1)
    return start, end


def _previous_session_name(current_session: str) -> str:
    order = ["TOKYO", "LONDON", "OVERLAP_LONDON_NY", "NEW_YORK"]
    if current_session not in order:
        return "NEW_YORK"
    idx = order.index(current_session)
    return order[idx - 1] if idx > 0 else order[-1]


def _load_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def _save_json(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _append_regime_history(entry: Dict[str, Any]) -> None:
    history = []
    try:
        if os.path.exists(REGIME_HISTORY_PATH):
            with open(REGIME_HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f) or []
    except Exception:
        history = []
    history.append(entry)
    try:
        with open(REGIME_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def _init_mt5(config: Dict[str, Any]) -> bool:
    mt5_cfg = config.get("mt5", {})
    path = mt5_cfg.get("path")
    try:
        mt5.shutdown()
    except Exception:
        pass
    return mt5.initialize(path) if path else mt5.initialize()


def _get_deals_between(config: Dict[str, Any], start: datetime, end: datetime) -> List[Any]:
    if not _init_mt5(config):
        return []
    try:
        deals = mt5.history_deals_get(start, end)
        if not deals:
            return []
        return [d for d in deals if getattr(d, "magic", None) == MAGIC_NUMBER]
    except Exception:
        return []
    finally:
        mt5.shutdown()


def _group_map(config: Dict[str, Any]) -> Dict[str, str]:
    groups = config.get("symbols", {}).get("groups", {})
    mapping: Dict[str, str] = {}
    for group_name, group_cfg in groups.items():
        symbols = group_cfg.get("symbols", []) if isinstance(group_cfg, dict) else []
        for sym in symbols:
            mapping[sym] = group_name.lower()
    return mapping


def _session_stats(deals: Iterable[Any], group_map: Dict[str, str]) -> Dict[str, Any]:
    metals_pnl = 0.0
    crypto_pnl = 0.0
    metals_wins = 0
    metals_losses = 0
    global_wins = []

    for deal in deals or []:
        symbol = getattr(deal, "symbol", "")
        group = group_map.get(symbol, "")
        entry = getattr(deal, "entry", None)
        profit = float(getattr(deal, "profit", 0.0) or 0.0)

        if entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY):
            if profit > 0:
                global_wins.append(profit)
                if group == "commodities" and symbol.upper().startswith(("XAU", "XAG", "XPT", "XPD")):
                    metals_wins += 1
            else:
                if group == "commodities" and symbol.upper().startswith(("XAU", "XAG", "XPT", "XPD")):
                    metals_losses += 1
            if group == "commodities" and symbol.upper().startswith(("XAU", "XAG", "XPT", "XPD")):
                metals_pnl += profit
            if group == "crypto":
                crypto_pnl += profit

    avg_win = sum(global_wins) / len(global_wins) if global_wins else 0.0
    return {
        "global": {"avg_win": avg_win},
        "metals": {"blue_trades": metals_wins, "avg_win": avg_win, "consecutive_losses": metals_losses},
        "crypto_net_pnl": crypto_pnl,
        "metals_net_pnl": metals_pnl,
    }


def observe_metals_impulse(config: Dict[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    state = _load_json(REGIME_STATE_PATH)
    current_session = _get_session_name(now)
    session_boundary = state.get("last_session") not in (None, current_session)

    state["last_session"] = current_session
    _save_json(REGIME_STATE_PATH, state)

    current_window = _get_session_window(current_session, now)
    last_session = _previous_session_name(current_session)
    last_window = _get_session_window(last_session, now)

    group_map = _group_map(config)
    current_deals = _get_deals_between(config, *current_window)
    last_deals = _get_deals_between(config, *last_window)

    current_stats = _session_stats(current_deals, group_map)
    last_stats = _session_stats(last_deals, group_map)

    current_stats["metals"]["first_red_after_impulse"] = False
    current_stats["session_boundary"] = session_boundary

    if not validate_metals_impulse({
        "metals": {
            "consecutive_losses": current_stats["metals"]["consecutive_losses"],
            "first_red_after_impulse": current_stats["metals"]["first_red_after_impulse"],
        },
        "session_boundary": session_boundary,
    }):
        return

    # Log observer entry if metals impulse is detected
    if last_stats["metals"]["blue_trades"] >= 2 and last_stats["metals"]["avg_win"] > current_stats["global"]["avg_win"]:
        duration_minutes = int((current_window[1] - current_window[0]).total_seconds() / 60)
        entry = {
            "regime": "METALS_IMPULSE",
            "duration_minutes": duration_minutes,
            "metals_net_pnl": round(current_stats["metals_net_pnl"], 2),
            "crypto_net_pnl": round(current_stats["crypto_net_pnl"], 2),
            "outcome": "POSITIVE" if current_stats["metals_net_pnl"] >= 0 else "NEGATIVE",
        }
        _append_regime_history(entry)
