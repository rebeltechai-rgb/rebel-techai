"""
REBEL Position Manager - Trade Tracking & Exit System
=====================================================
Tracks open positions, monitors heat/favorable excursions,
detects closures, and triggers ML label logging.

Now with PERSISTENCE: Tracked positions are saved to disk
so bot restarts don't lose position tracking.
"""

import os
import json
import MetaTrader5 as mt5
from datetime import datetime, timezone

import rebel_trade_limiter
from typing import Dict, List, Optional, Any

# Persistence file for tracked positions
TRACKED_POSITIONS_FILE = r"C:\Rebel Technologies\Rebel Master\Config\tracked_positions.json"


class TrackedPosition:
    """Represents a tracked position with metrics."""
    
    def __init__(self, ticket: int, symbol: str, direction: str, 
                 entry_price: float, volume: float, sl: float, tp: float,
                 open_time: datetime, atr: float = 0.0):
        self.ticket = ticket
        self.symbol = symbol
        self.direction = direction  # "BUY" or "SELL"
        self.entry_price = entry_price
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.open_time = open_time
        
        # ML-relevant metrics
        self.atr = atr  # ATR at entry for volatility normalization
        self.sl_distance = abs(entry_price - sl) if sl > 0 else 0.0
        
        # Tracking metrics
        self.max_heat = 0.0  # Max adverse excursion (MAE)
        self.max_favorable = 0.0  # Max favorable excursion (MFE)
        self.current_price = entry_price
        self.current_profit_points = 0.0
        
    def update(self, current_price: float):
        """Update position with current price and recalculate metrics."""
        self.current_price = current_price
        
        if self.direction == "BUY":
            self.current_profit_points = current_price - self.entry_price
        else:  # SELL
            self.current_profit_points = self.entry_price - current_price
        
        # Update max favorable (best profit seen)
        if self.current_profit_points > self.max_favorable:
            self.max_favorable = self.current_profit_points
        
        # Update max heat (worst drawdown seen) - stored as positive value
        if self.current_profit_points < 0:
            heat = abs(self.current_profit_points)
            if heat > self.max_heat:
                self.max_heat = heat
    
    def to_outcome_dict(self, exit_price: float, close_time: datetime) -> Dict[str, Any]:
        """Convert to outcome dict for ML label logging."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "max_heat": self.max_heat,
            "max_favorable": self.max_favorable,
            "open_time": self.open_time,
            "close_time": close_time,
            "volume": self.volume,
            "ticket": self.ticket
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "volume": self.volume,
            "sl": self.sl,
            "tp": self.tp,
            "open_time": self.open_time.isoformat(),
            "atr": self.atr,
            "sl_distance": self.sl_distance,
            "max_heat": self.max_heat,
            "max_favorable": self.max_favorable,
            "current_price": self.current_price,
            "current_profit_points": self.current_profit_points
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedPosition":
        """Deserialize from dictionary."""
        # Parse open_time and ensure it's timezone-aware
        open_time = datetime.fromisoformat(data["open_time"])
        if open_time.tzinfo is None:
            open_time = open_time.replace(tzinfo=timezone.utc)
        
        pos = cls(
            ticket=data["ticket"],
            symbol=data["symbol"],
            direction=data["direction"],
            entry_price=data["entry_price"],
            volume=data["volume"],
            sl=data["sl"],
            tp=data["tp"],
            open_time=open_time,
            atr=data.get("atr", 0.0)
        )
        pos.sl_distance = data.get("sl_distance", pos.sl_distance)
        pos.max_heat = data.get("max_heat", 0.0)
        pos.max_favorable = data.get("max_favorable", 0.0)
        pos.current_price = data.get("current_price", pos.entry_price)
        pos.current_profit_points = data.get("current_profit_points", 0.0)
        return pos


class RebelPositionManager:
    """
    Manages open positions, tracks metrics, detects closures,
    and triggers ML label logging.
    
    Now with PERSISTENCE: Positions are saved to disk so bot
    restarts don't lose tracking data (MFE, MAE, etc).
    """
    
    def __init__(self, ml_labeler=None):
        """
        Initialize position manager.
        
        Args:
            ml_labeler: MLLabelGenerator instance for logging outcomes
        """
        self.ml_labeler = ml_labeler
        self.tracked_positions: Dict[int, TrackedPosition] = {}  # ticket -> TrackedPosition
        self.closed_count = 0
        self.total_profit_points = 0.0
        self.ml_label_issue_log = r"C:\Rebel Technologies\Rebel Master\logs\ml_label_issues.log"
        
        # Load persisted positions on startup
        self._load_tracked_positions()
    
    def _ensure_config_dir(self):
        """Ensure config directory exists."""
        config_dir = os.path.dirname(TRACKED_POSITIONS_FILE)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

    def _ensure_logs_dir(self):
        """Ensure logs directory exists."""
        log_dir = os.path.dirname(self.ml_label_issue_log)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _log_label_issue(self, message: str) -> None:
        """Persist ML label logging issues for later diagnosis."""
        self._ensure_logs_dir()
        try:
            with open(self.ml_label_issue_log, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} | {message}\n")
        except Exception as e:
            print(f"[ML_LABEL] Failed to write label issue log: {e}")
    
    def _save_tracked_positions(self):
        """Save tracked positions to disk."""
        try:
            self._ensure_config_dir()
            data = {
                "tracked_positions": {
                    str(ticket): pos.to_dict() 
                    for ticket, pos in self.tracked_positions.items()
                },
                "closed_count": self.closed_count,
                "total_profit_points": self.total_profit_points,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            with open(TRACKED_POSITIONS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[POSITION MGR] Error saving tracked positions: {e}")
    
    def _load_tracked_positions(self):
        """Load tracked positions from disk on startup."""
        if not os.path.exists(TRACKED_POSITIONS_FILE):
            print("[POSITION MGR] No persisted positions file found, starting fresh")
            return
        
        try:
            with open(TRACKED_POSITIONS_FILE, "r") as f:
                data = json.load(f)
            
            positions_data = data.get("tracked_positions", {})
            loaded_count = 0
            
            for ticket_str, pos_data in positions_data.items():
                try:
                    ticket = int(ticket_str)
                    pos = TrackedPosition.from_dict(pos_data)
                    self.tracked_positions[ticket] = pos
                    loaded_count += 1
                except Exception as e:
                    print(f"[POSITION MGR] Error loading position {ticket_str}: {e}")
            
            self.closed_count = data.get("closed_count", 0)
            self.total_profit_points = data.get("total_profit_points", 0.0)
            
            if loaded_count > 0:
                print(f"[POSITION MGR] Loaded {loaded_count} tracked positions from disk")
                print(f"[POSITION MGR] Lifetime stats: {self.closed_count} closed, {self.total_profit_points:.5f} pts")
        except Exception as e:
            print(f"[POSITION MGR] Error loading tracked positions: {e}")
    
    def startup_sync(self):
        """
        Sync with MT5 on startup to handle positions opened/closed while bot was down.
        Call this after MT5 is initialized.
        """
        if mt5.terminal_info() is None:
            print("[POSITION MGR] MT5 not ready - startup sync skipped")
            return

        print("[POSITION MGR] Running startup sync with MT5...")
        
        # Get current MT5 positions
        positions = mt5.positions_get()
        if positions is None:
            positions = []
        
        current_mt5_tickets = {pos.ticket for pos in positions}
        tracked_tickets = set(self.tracked_positions.keys())
        
        # Find positions in MT5 but not tracked (new while bot was down)
        new_tickets = current_mt5_tickets - tracked_tickets
        
        # Find positions tracked but not in MT5 (closed while bot was down)
        closed_tickets = tracked_tickets - current_mt5_tickets
        
        # Add new positions
        for pos in positions:
            if pos.ticket in new_tickets:
                direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                atr = self._get_symbol_atr(pos.symbol)
                
                tracked = TrackedPosition(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    direction=direction,
                    entry_price=pos.price_open,
                    volume=pos.volume,
                    sl=pos.sl,
                    tp=pos.tp,
                    open_time=datetime.fromtimestamp(pos.time, tz=timezone.utc),
                    atr=atr
                )
                self.tracked_positions[pos.ticket] = tracked
                print(f"[POSITION MGR] Startup: Added {pos.symbol} {direction} #{pos.ticket}")
        
        # Handle closed positions (log labels if possible)
        for ticket in closed_tickets:
            print(f"[POSITION MGR] Startup: Position #{ticket} was closed while bot was down")
            # Try to log label from deal history
            self._handle_closed_position(ticket)
        
        if new_tickets or closed_tickets:
            self._save_tracked_positions()
            print(f"[POSITION MGR] Startup sync: {len(new_tickets)} added, {len(closed_tickets)} closed")
        else:
            print("[POSITION MGR] Startup sync: All positions in sync")
        
    def sync_positions(self) -> List[int]:
        """
        Sync with MT5 to detect new and closed positions.
        
        Returns:
            List of newly detected position tickets
        """
        new_tickets = []

        if mt5.terminal_info() is None:
            print("[POSITION MGR] MT5 not ready - skipping sync")
            return new_tickets
        
        # Get all current MT5 positions
        positions = mt5.positions_get()
        if positions is None:
            positions = []
        
        current_tickets = set()
        
        for pos in positions:
            ticket = pos.ticket
            current_tickets.add(ticket)
            
            # Check if this is a new position we're not tracking
            if ticket not in self.tracked_positions:
                # New position detected
                direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                
                # Get ATR for this symbol (for ML labels)
                atr = self._get_symbol_atr(pos.symbol)
                
                tracked = TrackedPosition(
                    ticket=ticket,
                    symbol=pos.symbol,
                    direction=direction,
                    entry_price=pos.price_open,
                    volume=pos.volume,
                    sl=pos.sl,
                    tp=pos.tp,
                    open_time=datetime.fromtimestamp(pos.time, tz=timezone.utc),
                    atr=atr
                )
                
                self.tracked_positions[ticket] = tracked
                new_tickets.append(ticket)
                print(f"[POSITION MGR] Tracking new position: {pos.symbol} {direction} #{ticket}")
        
        # Save if we added new positions
        if new_tickets:
            self._save_tracked_positions()
        
        # Detect closed positions
        closed_tickets = []
        for ticket in list(self.tracked_positions.keys()):
            if ticket not in current_tickets:
                closed_tickets.append(ticket)
        
        # Process closed positions
        for ticket in closed_tickets:
            self._handle_closed_position(ticket)
        
        # Save if any positions closed
        if closed_tickets:
            self._save_tracked_positions()
        
        return new_tickets
    
    def _handle_closed_position(self, ticket: int):
        """Handle a position that has been closed."""
        if ticket not in self.tracked_positions:
            return
        
        tracked = self.tracked_positions[ticket]
        close_time = datetime.now(timezone.utc)
        
        # Try to get actual exit price from deal history
        exit_price = self._get_exit_price(ticket, tracked.symbol)
        if exit_price is None:
            # Fallback to last tracked price
            exit_price = tracked.current_price
        
        # Calculate final profit
        if tracked.direction == "BUY":
            profit_points = exit_price - tracked.entry_price
        else:
            profit_points = tracked.entry_price - exit_price
        
        self.total_profit_points += profit_points
        self.closed_count += 1
        
        # Determine outcome
        if profit_points > 0:
            outcome = "WIN ✅"
        elif profit_points < 0:
            outcome = "LOSS ❌"
        else:
            outcome = "BREAK-EVEN ➖"

        # Calculate R:R for limiter tracking
        trade_rr = 0.0
        if tracked.sl_distance > 0:
            trade_rr = abs(profit_points / tracked.sl_distance)

        # Record outcome for cooldown tracking (with R:R)
        try:
            if profit_points > 0:
                rebel_trade_limiter.record_win(tracked.symbol, rr=trade_rr)
            elif profit_points < 0:
                rebel_trade_limiter.record_loss(tracked.symbol, rr=trade_rr)
            else:
                rebel_trade_limiter.record_break_even(tracked.symbol)
        except Exception as e:
            print(f"[LIMITER] Error recording outcome for {tracked.symbol}: {e}")
        
        print(f"[POSITION MGR] Position closed: {tracked.symbol} {tracked.direction} #{ticket}")
        print(f"               Entry: {tracked.entry_price:.5f} → Exit: {exit_price:.5f}")
        print(f"               P/L: {profit_points:.5f} pts | Max Heat: {tracked.max_heat:.5f} | Max Fav: {tracked.max_favorable:.5f}")
        print(f"               Result: {outcome}")
        
        # Log to ML labeler (Step 7 enhanced labels)
        if self.ml_labeler:
            try:
                # Calculate time in trade (minutes)
                time_in_trade = (close_time - tracked.open_time).total_seconds() / 60.0
                
                # Direction for ML: "long" or "short" (use .upper() for safety)
                ml_direction = "long" if tracked.direction.upper() == "BUY" else "short"
                
                self.ml_labeler.log_label(
                    ticket=tracked.ticket,
                    symbol=tracked.symbol,
                    direction=ml_direction,
                    entry_price=tracked.entry_price,
                    exit_price=exit_price,
                    sl_distance=tracked.sl_distance,
                    atr=tracked.atr,
                    mfe=tracked.max_favorable,
                    mae=tracked.max_heat,
                    time_in_trade=round(time_in_trade, 2)
                )
            except Exception as e:
                msg = f"ticket={tracked.ticket} symbol={tracked.symbol} error={e}"
                print(f"[ML_LABEL] Error logging label for {tracked.symbol}: {e}")
                self._log_label_issue(msg)
        
        # Remove from tracking
        del self.tracked_positions[ticket]
    
    def _get_symbol_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate ATR for a symbol for ML label normalization."""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 50)
            if rates is None or len(rates) < period + 1:
                return 0.0
            
            import pandas as pd
            df = pd.DataFrame(rates)
            
            high = df["high"]
            low = df["low"]
            close = df["close"]
            prev_close = close.shift(1)
            
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return float(atr.iloc[-1]) if len(atr) > 0 and pd.notna(atr.iloc[-1]) else 0.0
        except Exception as e:
            print(f"[POSITION MGR] ATR calc error for {symbol}: {e}")
            return 0.0
    
    def _get_exit_price(self, ticket: int, symbol: str) -> Optional[float]:
        """Try to get the actual exit price from deal history."""
        try:
            # Get deals for this position
            deals = mt5.history_deals_get(position=ticket)
            if deals and len(deals) > 0:
                # Find the closing deal (last deal with this position ID)
                for deal in reversed(deals):
                    if deal.entry == mt5.DEAL_ENTRY_OUT:  # Exit deal
                        return deal.price
            
            # Fallback: get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return tick.bid  # Use bid as approximation
        except Exception as e:
            print(f"[POSITION MGR] Could not get exit price for #{ticket}: {e}")
        
        return None
    
    def update_all_positions(self):
        """Update all tracked positions with current prices."""
        updated = False
        for ticket, tracked in self.tracked_positions.items():
            tick = mt5.symbol_info_tick(tracked.symbol)
            if tick:
                # Use appropriate price based on direction
                if tracked.direction == "BUY":
                    current_price = tick.bid  # Would close at bid
                else:
                    current_price = tick.ask  # Would close at ask
                
                old_mfe = tracked.max_favorable
                old_mae = tracked.max_heat
                tracked.update(current_price)
                
                # Track if MFE/MAE changed significantly
                if tracked.max_favorable > old_mfe or tracked.max_heat > old_mae:
                    updated = True
        
        # Save periodically when metrics change
        if updated:
            self._save_tracked_positions()
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of current positions."""
        total_heat = sum(p.max_heat for p in self.tracked_positions.values())
        total_favorable = sum(p.max_favorable for p in self.tracked_positions.values())
        total_current_profit = sum(p.current_profit_points for p in self.tracked_positions.values())
        
        return {
            "open_count": len(self.tracked_positions),
            "closed_count": self.closed_count,
            "total_current_profit_points": round(total_current_profit, 5),
            "total_heat": round(total_heat, 5),
            "total_favorable": round(total_favorable, 5),
            "lifetime_profit_points": round(self.total_profit_points, 5),
            "positions": [
                {
                    "ticket": t.ticket,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "current_profit": round(t.current_profit_points, 5),
                    "max_heat": round(t.max_heat, 5),
                    "max_favorable": round(t.max_favorable, 5)
                }
                for t in self.tracked_positions.values()
            ]
        }
    
    def manage_positions(self) -> Dict[str, Any]:
        """
        Main position management cycle.
        Call this periodically from the engine.
        
        Returns:
            Position summary dict
        """
        if mt5.terminal_info() is None:
            print("[POSITION MGR] MT5 not ready - skipping sync/update")
            return self.get_position_summary()

        # Sync with MT5 (detect new/closed positions)
        new_tickets = self.sync_positions()
        
        # Update all tracked positions with current prices
        self.update_all_positions()
        
        # Get summary
        summary = self.get_position_summary()
        
        if new_tickets:
            print(f"[POSITION MGR] New positions detected: {new_tickets}")
        
        if summary["open_count"] > 0:
            print(f"[POSITION MGR] Open: {summary['open_count']} | "
                  f"Current P/L: {summary['total_current_profit_points']:.5f} pts | "
                  f"Heat: {summary['total_heat']:.5f} | Favorable: {summary['total_favorable']:.5f}")
        
        return summary
    
    def close_position(self, ticket: int, reason: str = "Manual") -> bool:
        """
        Close a specific position.
        
        Args:
            ticket: Position ticket to close
            reason: Reason for closing
            
        Returns:
            True if close request was sent successfully
        """
        if ticket not in self.tracked_positions:
            print(f"[POSITION MGR] Position #{ticket} not found in tracking")
            return False
        
        tracked = self.tracked_positions[ticket]
        
        # Get current position info from MT5
        position = mt5.positions_get(ticket=ticket)
        if not position or len(position) == 0:
            print(f"[POSITION MGR] Position #{ticket} not found in MT5")
            return False
        
        pos = position[0]
        symbol_info = mt5.symbol_info(tracked.symbol)
        
        if symbol_info is None:
            print(f"[POSITION MGR] Symbol info not found for {tracked.symbol}")
            return False
        
        # Prepare close request
        if tracked.direction == "BUY":
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(tracked.symbol).bid
        else:
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(tracked.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": tracked.symbol,
            "volume": pos.volume,
            "type": trade_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 20251202,
            "comment": f"REBEL Close: {reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[POSITION MGR] Closed position #{ticket}: {reason}")
            return True
        else:
            error = result.retcode if result else "No result"
            print(f"[POSITION MGR] Failed to close #{ticket}: {error}")
            return False
    
    def close_all_positions(self, reason: str = "Close All") -> int:
        """
        Close all tracked positions.
        
        Returns:
            Number of positions closed
        """
        closed = 0
        for ticket in list(self.tracked_positions.keys()):
            if self.close_position(ticket, reason):
                closed += 1
        return closed

