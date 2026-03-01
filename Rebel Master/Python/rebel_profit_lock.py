"""
rebel_profit_lock.py

PROFIT-LOCK LADDER - Primary Exit Authority (DOLLAR-BASED)

Applies to ALL symbols (FX, XAU, Indices, Crypto, Softs).
No ATR dependency. No point calculations needed.

LADDER (in dollars):
    FX:     +$2 -> BE, +$5 -> +$2, +$10 -> +$5, +$15 -> +$8, +$20 -> +$10
    OTHER:  +$5 -> BE, +$15 -> +$5, +$25 -> +$12, +$40 -> +$20, +$60 -> +$30

CORE RULES:
    - SL only moves forward - never backward
    - Continuous evaluation - every tick / loop
    - Lock upgrades immediately when tier is crossed
    - If price touches lock -> close immediately
    - Once locked, trade can never go red

Overrides ATR-based SL behavior once active.
"""

import MetaTrader5 as mt5
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# =============================================================================
# PROFIT-LOCK LADDER (in DOLLARS)
# =============================================================================

# FX ladder - smaller dollar values (typical FX moves are smaller)
FX_LADDER = [
    # (trigger_dollars, lock_dollars)
    (2, 0.50),    # +$2 profit -> lock at BE (+$0.50 buffer)
    (5, 2),       # +$5 profit -> lock at +$2
    (10, 5),      # +$10 profit -> lock at +$5
    (15, 8),      # +$15 profit -> lock at +$8
    (20, 10),     # +$20 profit -> lock at +$10
    (30, 15),     # +$30 profit -> lock at +$15
    (50, 25),     # +$50 profit -> lock at +$25
]

# Standard ladder for Gold, Crypto, Indices, Energies, etc.
STANDARD_LADDER = [
    # (trigger_dollars, lock_dollars)
    (5, 1),       # +$5 profit -> lock at BE (+$1 buffer)
    (15, 5),      # +$15 profit -> lock at +$5
    (25, 12),     # +$25 profit -> lock at +$12
    (40, 20),     # +$40 profit -> lock at +$20
    (60, 30),     # +$60 profit -> lock at +$30
    (100, 50),    # +$100 profit -> lock at +$50
    (150, 80),    # +$150 profit -> lock at +$80
]

# Lock log path
LOCK_LOG_PATH = r"C:\Rebel Technologies\Rebel Master\logs\profit_lock.txt"

# Debug logging for specific symbols (logs to profit_lock.txt, throttled)
DEBUG_PROFIT_LOCK = True
DEBUG_SYMBOLS = {"XAGUSD", "NAS100", "USTECH"}


def _is_fx_symbol(symbol: str) -> bool:
    """Check if symbol is an FX pair."""
    sym = symbol.upper()
    base = sym.replace(".A", "").replace(".SA", "").replace(".FS", "").replace(".M", "")
    # FX pairs are 6 letters (EURUSD, GBPJPY, etc.)
    return len(base) == 6 and base.isalpha()


def get_ladder_for_symbol(symbol: str) -> list:
    """Get the appropriate dollar ladder for the symbol."""
    if _is_fx_symbol(symbol):
        return FX_LADDER
    return STANDARD_LADDER


def _log_lock_event(symbol: str, ticket: int, event: str, details: str):
    """Log profit lock events."""
    import os
    log_dir = os.path.dirname(LOCK_LOG_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {symbol} | #{ticket} | {event} | {details}\n"
    
    with open(LOCK_LOG_PATH, "a") as f:
        f.write(line)


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol names for matching (strip broker suffixes)."""
    return (
        symbol.upper()
        .replace(".FS", "")
        .replace(".A", "")
        .replace(".SA", "")
        .replace(".M", "")
    )


def get_position_profit_dollars(position: Dict[str, Any]) -> float:
    """
    Get current profit in dollars (from MT5 position data).
    
    Args:
        position: Position dict with 'profit' field
        
    Returns:
        Profit in dollars (positive = in profit, negative = in loss)
    """
    return position.get("profit", 0.0)


def calculate_lock_sl(
    position: Dict[str, Any],
    symbol: str
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate the new SL based on dollar profit-lock ladder.
    
    Args:
        position: Position dict
        symbol: Trading symbol
        
    Returns:
        Tuple of (new_sl_price, trigger_dollars, lock_dollars) or None if no lock needed
    """
    entry_price = position.get("price_open", 0)
    current_price = position.get("price_current", 0)
    current_sl = position.get("sl", 0)
    current_profit = get_position_profit_dollars(position)
    volume = position.get("volume", 0)
    pos_type = position.get("type", 0)
    
    # Handle string type
    if isinstance(pos_type, str):
        is_long = pos_type.lower() == "buy"
    else:
        is_long = pos_type == 0
    
    if current_profit <= 0:
        return None  # Not in profit, no lock
    
    if volume <= 0:
        return None  # Invalid volume
    
    # Get the right ladder for this symbol
    ladder = get_ladder_for_symbol(symbol)
    
    # Find the highest tier that is triggered
    active_tier = None
    for trigger_dollars, lock_dollars in ladder:
        if current_profit >= trigger_dollars:
            active_tier = (trigger_dollars, lock_dollars)
    
    if active_tier is None:
        return None  # Not enough profit to trigger any lock
    
    trigger_dollars, lock_dollars = active_tier
    
    # Get symbol info for price calculations
    info = mt5.symbol_info(symbol)
    if not info:
        return None
    
    point = info.point
    tick_value = info.trade_tick_value
    tick_size = info.trade_tick_size
    contract_size = info.trade_contract_size
    
    if tick_value <= 0 or volume <= 0:
        return None
    
    # Calculate price distance needed to lock the target profit
    # profit = (price_change / tick_size) * tick_value * volume
    # price_change = (lock_dollars / (tick_value * volume)) * tick_size
    
    # Price distance from entry to achieve lock_dollars profit
    lock_price_distance = (lock_dollars * tick_size) / (tick_value * volume)
    
    # Calculate new SL price
    if is_long:
        # For longs: SL is below entry, lock moves it up
        new_sl = entry_price + lock_price_distance
    else:
        # For shorts: SL is above entry, lock moves it down
        new_sl = entry_price - lock_price_distance
    
    # Round to symbol's precision
    digits = info.digits
    new_sl = round(new_sl, digits)
    
    # RULE: SL only moves forward, never backward
    if current_sl and current_sl > 0:
        if is_long and new_sl <= current_sl:
            return None  # Would loosen SL for long
        if not is_long and new_sl >= current_sl:
            return None  # Would loosen SL for short
    
    return (new_sl, trigger_dollars, lock_dollars)


def should_close_at_lock(position: Dict[str, Any], symbol: str) -> bool:
    """
    Check if price has touched or breached the lock level.
    If so, close immediately.
    
    Args:
        position: Position dict
        symbol: Trading symbol
        
    Returns:
        True if position should be closed
    """
    current_price = position.get("price_current", 0)
    current_sl = position.get("sl", 0)
    pos_type = position.get("type", 0)
    
    if current_sl <= 0:
        return False
    
    # Handle string type
    if isinstance(pos_type, str):
        is_long = pos_type.lower() == "buy"
    else:
        is_long = pos_type == 0
    
    if is_long:
        # For longs: close if price <= SL
        return current_price <= current_sl
    else:
        # For shorts: close if price >= SL
        return current_price >= current_sl


class ProfitLockManager:
    """
    Manages profit-lock ladder for all positions.
    
    Call update_all_positions() on every tick/loop to evaluate locks.
    """
    
    def __init__(self):
        self.lock_status: Dict[int, Dict] = {}  # ticket -> lock info
    
    def update_position(self, position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate and update profit lock for a single position.
        
        Args:
            position: Position dict from broker
            
        Returns:
            Dict with action info or None if no action needed
        """
        ticket = position.get("ticket", 0)
        symbol = position.get("symbol", "")
        
        if not ticket or not symbol:
            return None

        # Throttled debug logging for targeted symbols
        normalized = _normalize_symbol(symbol)
        if DEBUG_PROFIT_LOCK and normalized in DEBUG_SYMBOLS:
            status = self.lock_status.get(ticket, {})
            last_debug = status.get("last_debug")
            if not last_debug or (datetime.now() - last_debug).total_seconds() >= 30:
                current_profit = get_position_profit_dollars(position)
                volume = position.get("volume", 0)
                current_sl = position.get("sl", 0)
                current_price = position.get("price_current", 0)
                pos_type = position.get("type", 0)
                _log_lock_event(
                    symbol,
                    ticket,
                    "DEBUG",
                    f"profit={current_profit:.2f} vol={volume} sl={current_sl} price={current_price} type={pos_type}"
                )
                status["last_debug"] = datetime.now()
                self.lock_status[ticket] = status
        
        # Check if should close at lock
        if should_close_at_lock(position, symbol):
            _log_lock_event(symbol, ticket, "CLOSE_AT_LOCK", 
                          f"Price touched SL lock, closing immediately")
            return {
                "action": "CLOSE",
                "ticket": ticket,
                "symbol": symbol,
                "reason": "lock_touched"
            }
        
        # Calculate new lock SL
        result = calculate_lock_sl(position, symbol)
        
        if result is None:
            return None
        
        new_sl, trigger_dollars, lock_dollars = result
        
        # Check if this is an upgrade
        prev_lock = self.lock_status.get(ticket, {}).get("lock_dollars", 0)
        
        if lock_dollars > prev_lock:
            # Lock upgrade!
            _log_lock_event(symbol, ticket, "LOCK_UPGRADE",
                          f"+${trigger_dollars:.0f} triggered -> Lock at +${lock_dollars:.0f} (SL={new_sl:.5f})")
            print(f"[LOCK] {symbol} #{ticket}: +${trigger_dollars:.0f} -> lock +${lock_dollars:.0f}")
            
            self.lock_status[ticket] = {
                "trigger_dollars": trigger_dollars,
                "lock_dollars": lock_dollars,
                "new_sl": new_sl,
                "timestamp": datetime.now()
            }
            
            return {
                "action": "MODIFY_SL",
                "ticket": ticket,
                "symbol": symbol,
                "new_sl": new_sl,
                "trigger_dollars": trigger_dollars,
                "lock_dollars": lock_dollars,
                "reason": f"profit_lock_+${trigger_dollars:.0f}"
            }
        
        return None
    
    def update_all_positions(self, positions: list) -> list:
        """
        Evaluate profit locks for all positions.
        
        Args:
            positions: List of position dicts from broker
            
        Returns:
            List of action dicts for positions needing updates
        """
        actions = []
        
        for pos in positions:
            action = self.update_position(pos)
            if action:
                actions.append(action)
        
        return actions
    
    def execute_lock_actions(self, actions: list) -> list:
        """
        Execute the profit lock actions (modify SL or close).
        
        Args:
            actions: List of action dicts from update_all_positions()
            
        Returns:
            List of result dicts
        """
        results = []
        
        for action in actions:
            ticket = action.get("ticket")
            symbol = action.get("symbol")
            action_type = action.get("action")
            
            if action_type == "MODIFY_SL":
                new_sl = action.get("new_sl")
                result = self._modify_sl(ticket, symbol, new_sl)
                lock_dollars = action.get("lock_dollars", 0)
                results.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "action": "MODIFY_SL",
                    "success": result,
                    "new_sl": new_sl,
                    "lock_dollars": lock_dollars
                })
                
            elif action_type == "CLOSE":
                result = self._close_position(ticket, symbol)
                results.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "action": "CLOSE",
                    "success": result
                })
        
        return results
    
    def _modify_sl(self, ticket: int, symbol: str, new_sl: float) -> bool:
        """Modify position SL via MT5."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        
        pos = positions[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": pos.tp,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[LOCK] {symbol} #{ticket}: SL locked at {new_sl:.5f}")
            return True
        else:
            error = result.comment if result else "Unknown error"
            print(f"[LOCK] {symbol} #{ticket}: Failed to modify SL - {error}")
            return False
    
    def _close_position(self, ticket: int, symbol: str) -> bool:
        """Close position via MT5."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        
        pos = positions[0]
        
        # Determine close direction
        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "REBEL profit lock close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[LOCK] {symbol} #{ticket}: Closed at lock level")
            _log_lock_event(symbol, ticket, "CLOSED", "Position closed at lock level")
            return True
        else:
            error = result.comment if result else "Unknown error"
            print(f"[LOCK] {symbol} #{ticket}: Failed to close - {error}")
            return False
    
    def get_lock_status(self, ticket: int) -> Optional[Dict]:
        """Get current lock status for a ticket."""
        return self.lock_status.get(ticket)
    
    def clear_closed_positions(self, open_tickets: set):
        """Remove lock status for positions that are no longer open."""
        closed = [t for t in self.lock_status if t not in open_tickets]
        for t in closed:
            del self.lock_status[t]


# =============================================================================
# STATUS & VERIFICATION
# =============================================================================

def print_profit_lock_status():
    """Print profit-lock ladder configuration."""
    print("\n" + "=" * 55)
    print("  PROFIT-LOCK LADDER - DOLLAR BASED")
    print("=" * 55)
    print()
    print("  FX PAIRS (smaller thresholds):")
    for trigger, lock in FX_LADDER:
        if lock <= 1:
            print(f"    +${trigger:5.0f} profit -> Lock at BE (+${lock:.2f})")
        else:
            print(f"    +${trigger:5.0f} profit -> Lock at +${lock:.0f}")
    print()
    print("  OTHER (Gold, Crypto, Indices, etc.):")
    for trigger, lock in STANDARD_LADDER:
        if lock <= 1:
            print(f"    +${trigger:5.0f} profit -> Lock at BE (+${lock:.0f})")
        else:
            print(f"    +${trigger:5.0f} profit -> Lock at +${lock:.0f}")
    print()
    print("  RULES:")
    print("    [x] SL only moves forward - never backward")
    print("    [x] Continuous evaluation - every tick/loop")
    print("    [x] Lock upgrades immediately when tier crossed")
    print("    [x] If price touches lock -> close immediately")
    print("    [x] Once locked, trade can never go red")
    print()
    print("  Uses actual $ profit from MT5 - no point math needed!")
    print("=" * 55 + "\n")


# Print status on import
print_profit_lock_status()
