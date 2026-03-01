"""
rebel_trade_limiter.py

Trade Limiter & Cooldown System for REBEL Trading Bot.

Features:
- 300 second cooldown after a single loss (per symbol)
- 900 second cooldown after consecutive losses (per symbol)
- Per-asset-class daily trade limits (for balanced ML training)
- Weekly trade limit (optional)

Tracks losses and enforces cooldowns per symbol.
"""

import os
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Cooldown settings (in seconds)
COOLDOWN_AFTER_LOSS = 180          # 3 minutes after first loss
COOLDOWN_AFTER_CONSECUTIVE = 300   # 5 minutes after consecutive losses

# Weekly trade limit (DISABLED - set very high for ML training)
MAX_TRADES_PER_WEEK = 999999  # Effectively unlimited

# Persistence file
LIMITER_STATE_FILE = r"C:\Rebel Technologies\Rebel Master\logs\trade_limiter_state.json"
LIMITER_LOG_FILE = r"C:\Rebel Technologies\Rebel Master\logs\trade_limiter.txt"
CONFIG_PATH = r"C:\Rebel Technologies\Rebel Master\Config\master_config.yaml"

# Default per-asset-class limits (overridden by config)
DEFAULT_CLASS_LIMITS = {
    "forex": 15,
    "crypto": 7,
    "metals": 5,
    "indices": 4,
    "energies": 3,
    "softs": 2
}

# Per-symbol limits (overridden by config)
DEFAULT_MAX_ATTEMPTS_PER_SYMBOL = 30
DEFAULT_MIN_TRADES_FOR_WIN_RATE = 30
DEFAULT_MIN_WIN_RATE = 0.40

# Permanently blocked symbols (never trade these)
BLOCKED_SYMBOLS = {"NOKSEK"}


def _ensure_log_dir():
    """Ensure the Logs directory exists."""
    log_dir = os.path.dirname(LIMITER_STATE_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def _log_event(event: str, details: str):
    """Log limiter events."""
    _ensure_log_dir()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {event} | {details}\n"
    
    with open(LIMITER_LOG_FILE, "a") as f:
        f.write(line)


def _get_week_start() -> datetime:
    """Get the start of the current week (Monday 00:00)."""
    now = datetime.now()
    days_since_monday = now.weekday()
    week_start = now - timedelta(days=days_since_monday)
    return week_start.replace(hour=0, minute=0, second=0, microsecond=0)


def _get_day_start() -> datetime:
    """Get the start of the current day (00:00 UTC)."""
    now = datetime.utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)

def _get_session_bucket(now: Optional[datetime] = None) -> str:
    """Return session bucket based on UTC hour."""
    current = now or datetime.utcnow()
    hour = current.hour
    if 7 <= hour < 16:
        return "london"
    if 13 <= hour < 22:
        return "newyork"
    if 0 <= hour < 9:
        return "asian"
    return "off_hours"

def _load_class_limits() -> Dict[str, int]:
    """Load per-asset-class limits from config."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = yaml.safe_load(f)
            limits = config.get("safety", {}).get("asset_class_limits", {})
            if limits:
                return {k.lower(): int(v) for k, v in limits.items()}
    except Exception as e:
        print(f"[LIMITER] Error loading class limits: {e}")
    return DEFAULT_CLASS_LIMITS.copy()


def _load_limiter_rules() -> Dict[str, float]:
    """Load per-symbol limiter rules from config."""
    rules = {
        "max_attempts_per_symbol": DEFAULT_MAX_ATTEMPTS_PER_SYMBOL,
        "min_trades_for_win_rate": DEFAULT_MIN_TRADES_FOR_WIN_RATE,
        "min_win_rate": DEFAULT_MIN_WIN_RATE,
    }
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = yaml.safe_load(f)
            cfg = config.get("trade_limiter", {}) if config else {}
            if cfg:
                rules["max_attempts_per_symbol"] = int(cfg.get("max_attempts_per_symbol", rules["max_attempts_per_symbol"]))
                rules["min_trades_for_win_rate"] = int(cfg.get("min_trades_for_win_rate", rules["min_trades_for_win_rate"]))
                rules["min_win_rate"] = float(cfg.get("min_win_rate", rules["min_win_rate"]))
    except Exception as e:
        print(f"[LIMITER] Error loading limiter rules: {e}")
    return rules


def _get_asset_group(symbol: str) -> str:
    """Classify symbol into asset group (duplicated from engine for isolation)."""
    s = symbol.upper()
    
    # Crypto
    crypto_tokens = ("BTC", "ETH", "XRP", "LTC", "ADA", "DOG", "DOT", "XLM", "SOL",
                     "AVAX", "AAVE", "BNB", "SAND", "UNI", "XTZ", "BCH", "COMP",
                     "CRV", "KSM", "LNK", "LRC", "MANA", "SUSHI", "BAT")
    if any(p in s for p in crypto_tokens):
        return "crypto"
    
    # Indices
    if any(k in s for k in ("US500", "US30", "US2000", "USTECH", "NAS100", "DAX40",
                            "SPA35", "UK100", "HK50", "CHINA50", "AUS200", "EU50",
                            "FRA40", "JPN225", "NETH25", "SWI20", "VIX", "USDINDEX",
                            "GER40", "IT40", "SGFREE", "CAC40", "EUSTX50", "HSI",
                            "NK225", "DJ30", "FT100", "SPI200", "S&P", "CN50")):
        return "indices"
    
    # Energies
    if any(k in s for k in ("UKOIL", "USOIL", "BRENT", "WTI", "NATGAS", "OIL", "GAS")):
        return "energies"
    
    # Metals
    if any(k in s for k in ("XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER", "COPPER")):
        return "metals"
    
    # Softs
    if any(k in s for k in ("COCOA", "COFFEE", "SOYBEAN", "SUGAR", "COTTON", "WHEAT", "CORN")):
        return "softs"
    
    # Default to forex
    return "forex"


# =============================================================================
# TRADE LIMITER CLASS
# =============================================================================

class TradeLimiter:
    """
    Manages trade cooldowns, weekly limits, and per-asset-class daily limits.
    
    Usage:
        limiter = TradeLimiter()
        
        # Before opening a trade:
        can_trade, reason = limiter.can_open_trade(symbol="EURUSD")
        if not can_trade:
            print(f"Trade blocked: {reason}")
            return
        
        # After trade closes:
        limiter.record_trade_result(won=False)  # or won=True
    """
    
    def __init__(self):
        self.class_limits = _load_class_limits()
        self.limiter_rules = _load_limiter_rules()
        self.state = self._load_state()
        self._cleanup_old_week()
        self._cleanup_old_day()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or create default."""
        _ensure_log_dir()
        
        if os.path.exists(LIMITER_STATE_FILE):
            try:
                with open(LIMITER_STATE_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        
        return self._default_state()
    
    def _default_state(self) -> Dict[str, Any]:
        """Create default state."""
        return {
            "week_start": _get_week_start().isoformat(),
            "day_start": _get_day_start().isoformat(),
            "trades_this_week": 0,
            "trades_today_by_class": {
                "forex": 0,
                "crypto": 0,
                "metals": 0,
                "indices": 0,
                "energies": 0,
                "softs": 0
            },
            "trades_today_total": 0,
            "consecutive_losses": 0,  # legacy global value (unused)
            "last_loss_time": None,   # legacy global value (unused)
            "cooldown_until": None,   # legacy global value (unused)
            "symbol_stats": {},
            "total_wins": 0,
            "total_losses": 0,
        }

    def _get_symbol_state(self, symbol: str) -> Dict[str, Any]:
        """Get or create per-symbol cooldown state."""
        if not symbol:
            return {}
        if "symbol_stats" not in self.state or not isinstance(self.state["symbol_stats"], dict):
            self.state["symbol_stats"] = {}
        key = symbol.upper().strip()
        if key not in self.state["symbol_stats"]:
            self.state["symbol_stats"][key] = {
                "consecutive_losses": 0,
                "last_loss_time": None,
                "cooldown_until": None,
                "last_session": None,
                "last_session_day": None,
                "session_block_session": None,
                "session_block_day": None,
                "wins": 0,
                "losses": 0,
                "attempts": 0,
                "total_rr": 0.0,
            }
        else:
            # Backfill new fields for existing state
            self.state["symbol_stats"][key].setdefault("wins", 0)
            self.state["symbol_stats"][key].setdefault("losses", 0)
            self.state["symbol_stats"][key].setdefault("attempts", 0)
            self.state["symbol_stats"][key].setdefault("total_rr", 0.0)
        return self.state["symbol_stats"][key]
    
    def _save_state(self):
        """Save state to file."""
        _ensure_log_dir()
        with open(LIMITER_STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def _cleanup_old_week(self):
        """Reset weekly counter if we're in a new week."""
        current_week_start = _get_week_start()
        saved_week_start = datetime.fromisoformat(self.state["week_start"])
        
        if current_week_start > saved_week_start:
            # New week - reset weekly counter
            _log_event("WEEK_RESET", f"New week started. Previous week: {self.state['trades_this_week']} trades")
            self.state["week_start"] = current_week_start.isoformat()
            self.state["trades_this_week"] = 0
            self._save_state()
    
    def _cleanup_old_day(self):
        """Reset daily counters if we're in a new day."""
        current_day_start = _get_day_start()
        saved_day_start_str = self.state.get("day_start")
        
        # Handle old state format without day_start
        if not saved_day_start_str:
            self.state["day_start"] = current_day_start.isoformat()
            self.state["trades_today_by_class"] = {k: 0 for k in self.class_limits}
            self.state["trades_today_total"] = 0
            self._save_state()
            return
        
        saved_day_start = datetime.fromisoformat(saved_day_start_str)
        
        if current_day_start > saved_day_start:
            # New day - reset daily counters AND per-symbol attempt counters
            summary = ", ".join(f"{k}:{v}" for k, v in self.state.get("trades_today_by_class", {}).items() if v > 0)
            _log_event("DAY_RESET", f"New day. Yesterday: {self.state.get('trades_today_total', 0)} trades [{summary}]")
            self.state["day_start"] = current_day_start.isoformat()
            self.state["trades_today_by_class"] = {k: 0 for k in self.class_limits}
            self.state["trades_today_total"] = 0
            # Reset attempt counters daily (not permanent blocks)
            for sym_key, sym_state in self.state.get("symbol_stats", {}).items():
                sym_state["attempts"] = 0
            _log_event("DAY_RESET", "All symbol attempt counters reset to 0")
            self._save_state()
    
    def can_open_trade(self, symbol: str = "") -> Tuple[bool, str]:
        """
        Check if a new trade can be opened.
        
        Args:
            symbol: Trading symbol (used to determine asset class)
        
        Returns:
            (can_trade: bool, reason: str)
        """
        self._cleanup_old_week()
        self._cleanup_old_day()
        
        # Permanent block list
        if symbol and symbol.upper().strip() in BLOCKED_SYMBOLS:
            reason = f"permanently_blocked_{symbol}"
            _log_event("BLOCKED", reason)
            return False, reason
        
        # Check weekly limit
        if self.state["trades_this_week"] >= MAX_TRADES_PER_WEEK:
            reason = f"weekly_limit_reached_{self.state['trades_this_week']}/{MAX_TRADES_PER_WEEK}"
            _log_event("BLOCKED", reason)
            return False, reason
        
        # Check per-asset-class daily limit
        if symbol:
            asset_class = _get_asset_group(symbol)
            class_limit = self.class_limits.get(asset_class, 999)
            
            # Ensure trades_today_by_class has this key
            if "trades_today_by_class" not in self.state:
                self.state["trades_today_by_class"] = {k: 0 for k in self.class_limits}
            
            class_count = self.state["trades_today_by_class"].get(asset_class, 0)
            
            if class_count >= class_limit:
                reason = f"daily_class_limit_{asset_class}_{class_count}/{class_limit}"
                _log_event("BLOCKED", f"{symbol}: {reason}")
                return False, reason
        
        # Check per-symbol attempt cap and win-rate floor
        if symbol:
            symbol_state = self._get_symbol_state(symbol)
            max_attempts = int(self.limiter_rules.get("max_attempts_per_symbol", DEFAULT_MAX_ATTEMPTS_PER_SYMBOL))
            attempts = int(symbol_state.get("attempts", 0) or 0)
            if max_attempts > 0 and attempts >= max_attempts:
                reason = f"symbol_attempt_cap_{symbol}_{attempts}/{max_attempts}"
                _log_event("BLOCKED", reason)
                return False, reason

            min_trades = int(self.limiter_rules.get("min_trades_for_win_rate", DEFAULT_MIN_TRADES_FOR_WIN_RATE))
            wins = int(symbol_state.get("wins", 0) or 0)
            losses = int(symbol_state.get("losses", 0) or 0)
            total = wins + losses
            total_rr = float(symbol_state.get("total_rr", 0.0) or 0.0)
            if total >= min_trades and total_rr < 0:
                win_rate = wins / total if total else 0.0
                reason = f"symbol_negative_rr_{symbol}_rr={total_rr:.2f}_wr={win_rate:.0%}_after_{total}_trades"
                _log_event("BLOCKED", reason)
                return False, reason

        # Check per-symbol session block (after 3 losses)
        if symbol:
            symbol_state = self._get_symbol_state(symbol)
            current_session = _get_session_bucket()
            current_day = _get_day_start().date().isoformat()
            blocked_session = symbol_state.get("session_block_session")
            blocked_day = symbol_state.get("session_block_day")
            if blocked_session == current_session and blocked_day == current_day:
                reason = f"session_loss_limit_{symbol}_{current_session}"
                return False, reason
        
        # Check per-symbol cooldown
        if symbol:
            symbol_state = self._get_symbol_state(symbol)
            cooldown_until = symbol_state.get("cooldown_until")
            if cooldown_until:
                cooldown_end = datetime.fromisoformat(cooldown_until)
                now = datetime.now()
                if now < cooldown_end:
                    remaining = (cooldown_end - now).total_seconds()
                    reason = f"cooldown_active_{symbol}_{remaining:.0f}s_remaining"
                    return False, reason
                # Cooldown expired
                symbol_state["cooldown_until"] = None
                self._save_state()
        elif self.state.get("cooldown_until"):
            # Legacy global cooldown fallback (should not be used)
            cooldown_end = datetime.fromisoformat(self.state["cooldown_until"])
            now = datetime.now()
            if now < cooldown_end:
                remaining = (cooldown_end - now).total_seconds()
                reason = f"cooldown_active_{remaining:.0f}s_remaining"
                return False, reason
            self.state["cooldown_until"] = None
            self._save_state()
        
        return True, "allowed"

    def record_filter_passed(self, symbol: str = ""):
        """Record a trade attempt after filters passed."""
        if not symbol:
            return
        symbol_state = self._get_symbol_state(symbol)
        symbol_state["attempts"] = int(symbol_state.get("attempts", 0) or 0) + 1
        _log_event("ATTEMPT", f"{symbol} attempts={symbol_state['attempts']}")
        self._save_state()
    
    def record_trade_opened(self, symbol: str = ""):
        """Record that a trade was opened (increment counters)."""
        self._cleanup_old_week()
        self._cleanup_old_day()
        
        # Increment weekly counter
        self.state["trades_this_week"] += 1
        
        # Increment daily counters
        if "trades_today_by_class" not in self.state:
            self.state["trades_today_by_class"] = {k: 0 for k in self.class_limits}
        if "trades_today_total" not in self.state:
            self.state["trades_today_total"] = 0
        
        self.state["trades_today_total"] += 1
        
        if symbol:
            asset_class = _get_asset_group(symbol)
            if asset_class not in self.state["trades_today_by_class"]:
                self.state["trades_today_by_class"][asset_class] = 0
            self.state["trades_today_by_class"][asset_class] += 1
            symbol_state = self._get_symbol_state(symbol)
            symbol_state["last_session"] = _get_session_bucket()
            symbol_state["last_session_day"] = _get_day_start().date().isoformat()
            
            class_limit = self.class_limits.get(asset_class, 999)
            class_count = self.state["trades_today_by_class"][asset_class]
            
            _log_event("TRADE_OPENED", 
                       f"{symbol} ({asset_class}) | Class: {class_count}/{class_limit} | "
                       f"Daily: {self.state['trades_today_total']} | Weekly: {self.state['trades_this_week']}")
        else:
            _log_event("TRADE_OPENED", f"Weekly count: {self.state['trades_this_week']}/{MAX_TRADES_PER_WEEK}")
        
        self._save_state()
    
    def record_trade_result(self, won: bool, symbol: str = "", rr: float = 0.0):
        """
        Record trade result and apply cooldown if needed.
        
        Args:
            won: True if trade was profitable, False if loss
            symbol: Trading symbol (for logging)
            rr: Reward-to-risk ratio of the trade (positive for wins, negative for losses)
        """
        if won:
            # Win - reset consecutive losses for this symbol
            if symbol:
                symbol_state = self._get_symbol_state(symbol)
                symbol_state["consecutive_losses"] = 0
                symbol_state["cooldown_until"] = None
                symbol_state["session_block_session"] = None
                symbol_state["session_block_day"] = None
                symbol_state["wins"] = int(symbol_state.get("wins", 0)) + 1
                symbol_state["total_rr"] = float(symbol_state.get("total_rr", 0.0) or 0.0) + abs(rr)
            self.state["total_wins"] += 1
            _log_event("WIN", f"{symbol} | rr=+{abs(rr):.2f} | Symbol loss streak reset")
        else:
            # Loss - increment and apply cooldown for this symbol
            if symbol:
                symbol_state = self._get_symbol_state(symbol)
                symbol_state["consecutive_losses"] += 1
                symbol_state["last_loss_time"] = datetime.now().isoformat()
                symbol_state["losses"] = int(symbol_state.get("losses", 0)) + 1
                symbol_state["total_rr"] = float(symbol_state.get("total_rr", 0.0) or 0.0) - abs(rr)
                loss_streak = symbol_state["consecutive_losses"]
            else:
                # Fallback for unknown symbol
                self.state["consecutive_losses"] = self.state.get("consecutive_losses", 0) + 1
                self.state["last_loss_time"] = datetime.now().isoformat()
                loss_streak = self.state["consecutive_losses"]
            
            self.state["total_losses"] += 1

            # Block the symbol for the rest of the session after 3 losses
            if symbol and loss_streak >= 3:
                symbol_state["session_block_session"] = _get_session_bucket()
                symbol_state["session_block_day"] = _get_day_start().date().isoformat()
            
            # Determine cooldown duration
            if loss_streak >= 2:
                cooldown_seconds = COOLDOWN_AFTER_CONSECUTIVE
                cooldown_type = "consecutive"
            else:
                cooldown_seconds = COOLDOWN_AFTER_LOSS
                cooldown_type = "single"
            
            cooldown_end = datetime.now() + timedelta(seconds=cooldown_seconds)
            if symbol:
                symbol_state["cooldown_until"] = cooldown_end.isoformat()
            else:
                self.state["cooldown_until"] = cooldown_end.isoformat()
            
            _log_event("LOSS", f"{symbol} | Consecutive: {loss_streak} | "
                      f"Cooldown: {cooldown_seconds}s ({cooldown_type})")
        
        self._save_state()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current limiter status."""
        self._cleanup_old_week()
        self._cleanup_old_day()
        
        # Build per-class status
        class_status = {}
        for asset_class, limit in self.class_limits.items():
            count = self.state.get("trades_today_by_class", {}).get(asset_class, 0)
            class_status[asset_class] = {
                "count": count,
                "limit": limit,
                "remaining": max(0, limit - count)
            }
        
        # Build cooldown summary (per-symbol)
        cooldown_active_symbols = []
        now = datetime.now()
        for sym, sym_state in self.state.get("symbol_stats", {}).items():
            cooldown_until = sym_state.get("cooldown_until")
            if not cooldown_until:
                continue
            cooldown_end = datetime.fromisoformat(cooldown_until)
            if now < cooldown_end:
                cooldown_active_symbols.append({
                    "symbol": sym,
                    "remaining_seconds": (cooldown_end - now).total_seconds()
                })
        
        max_cooldown_remaining = 0
        if cooldown_active_symbols:
            max_cooldown_remaining = max(s["remaining_seconds"] for s in cooldown_active_symbols)
        
        status = {
            "trades_this_week": self.state["trades_this_week"],
            "max_trades_per_week": MAX_TRADES_PER_WEEK,
            "trades_remaining": MAX_TRADES_PER_WEEK - self.state["trades_this_week"],
            "trades_today_total": self.state.get("trades_today_total", 0),
            "trades_today_by_class": class_status,
            "consecutive_losses": self.state.get("consecutive_losses", 0),
            "cooldown_active": len(cooldown_active_symbols) > 0,
            "cooldown_remaining_seconds": max_cooldown_remaining,
            "cooldown_active_symbols": cooldown_active_symbols,
            "total_wins": self.state["total_wins"],
            "total_losses": self.state["total_losses"],
        }
        
        return status
    
    def force_reset_cooldown(self):
        """Force reset the cooldown (manual override)."""
        self.state["cooldown_until"] = None
        self.state["consecutive_losses"] = 0
        self._save_state()
        _log_event("MANUAL_RESET", "Cooldown force reset by operator")
    
    def force_reset_weekly(self):
        """Force reset the weekly counter (manual override)."""
        self.state["trades_this_week"] = 0
        self._save_state()
        _log_event("MANUAL_RESET", "Weekly counter force reset by operator")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Create a global instance for easy access
_limiter_instance: Optional[TradeLimiter] = None


def get_limiter() -> TradeLimiter:
    """Get the global TradeLimiter instance."""
    global _limiter_instance
    if _limiter_instance is None:
        _limiter_instance = TradeLimiter()
    return _limiter_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def can_trade(symbol: str = "") -> Tuple[bool, str]:
    """Check if trading is allowed (convenience function)."""
    return get_limiter().can_open_trade(symbol)


def record_trade_opened(symbol: str = ""):
    """Record that a trade was opened (convenience function)."""
    get_limiter().record_trade_opened(symbol)

def record_filter_passed(symbol: str = ""):
    """Record a filter-passed attempt (convenience function)."""
    get_limiter().record_filter_passed(symbol)


def record_win(symbol: str = "", rr: float = 0.0):
    """Record a winning trade (convenience function)."""
    get_limiter().record_trade_result(won=True, symbol=symbol, rr=rr)


def record_loss(symbol: str = "", rr: float = 0.0):
    """Record a losing trade (convenience function)."""
    get_limiter().record_trade_result(won=False, symbol=symbol, rr=rr)


def record_break_even(symbol: str = ""):
    """Record a break-even trade (reset cooldown and loss streak)."""
    limiter = get_limiter()
    if symbol:
        symbol_state = limiter._get_symbol_state(symbol)
        symbol_state["consecutive_losses"] = 0
        symbol_state["cooldown_until"] = None
    else:
        limiter.state["consecutive_losses"] = 0
        limiter.state["cooldown_until"] = None
    limiter._save_state()
    _log_event("BREAKEVEN", f"{symbol} | Symbol loss streak reset")


def get_status() -> Dict[str, Any]:
    """Get current limiter status (convenience function)."""
    return get_limiter().get_status()


# =============================================================================
# STATUS DISPLAY
# =============================================================================

def print_limiter_status():
    """Print trade limiter status."""
    status = get_status()
    
    print("\n" + "=" * 60)
    print("  TRADE LIMITER — STATUS")
    print("=" * 60)
    print(f"  Weekly Trades: {status['trades_this_week']}/{status['max_trades_per_week']}")
    print(f"  Today Total: {status['trades_today_total']}")
    print()
    print("  TODAY BY ASSET CLASS:")
    for asset_class, info in status['trades_today_by_class'].items():
        bar = "█" * info['count'] + "░" * info['remaining']
        print(f"    {asset_class.upper():8s}: {info['count']:2d}/{info['limit']:2d} [{bar}]")
    print()
    print(f"  Consecutive Losses: {status['consecutive_losses']}")
    cooldown_count = len(status.get("cooldown_active_symbols", []))
    print(f"  Cooldown Active: {'YES' if status['cooldown_active'] else 'NO'}")
    if status['cooldown_active']:
        print(f"  Cooldown Remaining (max): {status['cooldown_remaining_seconds']:.0f}s | Symbols: {cooldown_count}")
    print()
    print(f"  Total Wins: {status['total_wins']}")
    print(f"  Total Losses: {status['total_losses']}")
    print()
    print("  RULES:")
    print(f"    Single loss cooldown: {COOLDOWN_AFTER_LOSS}s (5 min) per symbol")
    print(f"    Consecutive loss cooldown: {COOLDOWN_AFTER_CONSECUTIVE}s (15 min) per symbol")
    print("=" * 60 + "\n")


# Print status on import
print_limiter_status()

