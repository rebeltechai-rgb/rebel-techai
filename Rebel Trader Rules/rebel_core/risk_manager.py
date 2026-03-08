"""
REBEL RULES-BASED AI TRADING SYSTEM
Module: Risk Manager
Enforces all risk limits and position management
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta


class RiskManager:
    """
    Risk management engine.
    Enforces limits before any trade is placed.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Global limits
        self.max_total_trades = self.config.get('max_total_trades', 6)
        self.max_drawdown_percent = self.config.get('max_drawdown_percent', 6.0)
        self.max_daily_trades = self.config.get('max_daily_trades', 20)
        self.max_risk_per_loop = self.config.get('max_risk_per_loop', 2.0)
        
        # Group limits
        self.group_limits = self.config.get('group_limits', {})
        
        # Tracking
        self.daily_trade_count = 0
        self.daily_reset_date = None
        self.loop_risk_used = 0.0
    
    def can_trade(
        self,
        symbol: str,
        group: str,
        open_positions: List[Dict[str, Any]],
        account_info: Dict[str, Any],
        risk_scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Check if a trade is allowed under current risk rules.
        
        Args:
            symbol: Trading symbol
            group: Symbol group (e.g., "majors", "crypto")
            open_positions: List of current open positions
            account_info: Account balance/equity info
            risk_scale: Proposed risk scale for this trade
        
        Returns:
            Dictionary with allowed status and reason
        """
        # Reset daily counter if new day
        self._check_daily_reset()
        
        # === CHECK 1: Drawdown ===
        drawdown = self._calculate_drawdown(account_info)
        if drawdown >= self.max_drawdown_percent:
            return {
                "allowed": False,
                "reason": f"Drawdown {drawdown:.2f}% exceeds limit {self.max_drawdown_percent}%",
                "action": "EMERGENCY_CLOSE"
            }
        
        # Warn if approaching limit
        drawdown_warning = drawdown >= (self.max_drawdown_percent * 0.7)
        
        # === CHECK 2: Total Open Trades ===
        total_open = len(open_positions)
        if total_open >= self.max_total_trades:
            return {
                "allowed": False,
                "reason": f"Max open trades ({self.max_total_trades}) reached",
                "action": "WAIT"
            }
        
        # === CHECK 3: Group Limits ===
        group_limit = self.group_limits.get(group, {}).get('max_trades', 2)
        group_count = sum(1 for p in open_positions if self._symbol_in_group(p['symbol'], group))
        
        if group_count >= group_limit:
            return {
                "allowed": False,
                "reason": f"Group {group} limit ({group_limit}) reached",
                "action": "WAIT"
            }
        
        # === CHECK 4: Daily Trade Limit ===
        if self.daily_trade_count >= self.max_daily_trades:
            return {
                "allowed": False,
                "reason": f"Daily trade limit ({self.max_daily_trades}) reached",
                "action": "WAIT_TOMORROW"
            }
        
        # === CHECK 5: Loop Risk Budget ===
        if self.loop_risk_used + risk_scale > self.max_risk_per_loop:
            return {
                "allowed": False,
                "reason": f"Loop risk budget ({self.max_risk_per_loop}) would be exceeded",
                "action": "WAIT_NEXT_LOOP"
            }
        
        # === CHECK 6: Same Symbol Already Open ===
        symbol_open = any(p['symbol'] == symbol for p in open_positions)
        if symbol_open and not self.config.get('allow_same_symbol', False):
            return {
                "allowed": False,
                "reason": f"Already have open position on {symbol}",
                "action": "WAIT"
            }
        
        # === ALL CHECKS PASSED ===
        return {
            "allowed": True,
            "reason": "All risk checks passed",
            "warnings": ["Approaching drawdown limit"] if drawdown_warning else [],
            "drawdown": drawdown,
            "open_trades": total_open,
            "group_trades": group_count
        }
    
    def register_trade(self, risk_scale: float = 1.0) -> None:
        """Register that a trade was placed (for tracking)."""
        self.daily_trade_count += 1
        self.loop_risk_used += risk_scale
    
    def reset_loop(self) -> None:
        """Reset loop-level risk tracking (call at start of each scan loop)."""
        self.loop_risk_used = 0.0
    
    def get_risk_status(
        self,
        open_positions: List[Dict[str, Any]],
        account_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get current risk status summary.
        
        Returns:
            Dictionary with risk metrics
        """
        drawdown = self._calculate_drawdown(account_info)
        total_open = len(open_positions)
        total_profit = sum(p.get('profit', 0) for p in open_positions)
        
        # Group breakdown
        group_counts = {}
        for pos in open_positions:
            group = self._get_symbol_group(pos['symbol'])
            group_counts[group] = group_counts.get(group, 0) + 1
        
        # Risk level
        if drawdown >= self.max_drawdown_percent:
            risk_level = "CRITICAL"
        elif drawdown >= self.max_drawdown_percent * 0.7:
            risk_level = "HIGH"
        elif drawdown >= self.max_drawdown_percent * 0.4:
            risk_level = "ELEVATED"
        else:
            risk_level = "NORMAL"
        
        return {
            "drawdown": drawdown,
            "drawdown_limit": self.max_drawdown_percent,
            "risk_level": risk_level,
            "open_trades": total_open,
            "max_trades": self.max_total_trades,
            "daily_trades": self.daily_trade_count,
            "max_daily": self.max_daily_trades,
            "total_profit": total_profit,
            "group_breakdown": group_counts,
            "balance": account_info.get('balance', 0),
            "equity": account_info.get('equity', 0)
        }
    
    def should_emergency_close(self, account_info: Dict[str, Any]) -> bool:
        """Check if emergency close is needed."""
        drawdown = self._calculate_drawdown(account_info)
        return drawdown >= self.max_drawdown_percent
    
    def get_adjusted_risk_scale(
        self,
        base_risk_scale: float,
        account_info: Dict[str, Any],
        group: str
    ) -> float:
        """
        Adjust risk scale based on current conditions.
        
        Rules:
        - Reduce risk as drawdown increases
        - Apply group-specific multipliers
        - Cap at maximum allowed
        """
        drawdown = self._calculate_drawdown(account_info)
        
        # Drawdown-based reduction
        if drawdown >= self.max_drawdown_percent * 0.7:
            drawdown_mult = 0.5  # Half size near limit
        elif drawdown >= self.max_drawdown_percent * 0.5:
            drawdown_mult = 0.75
        else:
            drawdown_mult = 1.0
        
        # Group multiplier
        group_mult = self.group_limits.get(group, {}).get('risk_scale', 1.0)
        
        # Calculate final
        final_scale = base_risk_scale * drawdown_mult * group_mult
        
        # Cap at 1.5
        return min(1.5, max(0.3, final_scale))
    
    def _calculate_drawdown(self, account_info: Dict[str, Any]) -> float:
        """Calculate current drawdown percentage."""
        balance = account_info.get('balance', 0)
        equity = account_info.get('equity', 0)
        
        if balance <= 0:
            return 0.0
        
        drawdown = ((balance - equity) / balance) * 100
        return max(0.0, drawdown)
    
    def _check_daily_reset(self) -> None:
        """Reset daily counter at midnight UTC."""
        today = datetime.now(timezone.utc).date()
        
        if self.daily_reset_date != today:
            self.daily_trade_count = 0
            self.daily_reset_date = today
    
    def _symbol_in_group(self, symbol: str, group: str) -> bool:
        """Check if symbol belongs to group."""
        group_symbols = self.config.get('groups', {}).get(group, [])
        
        # Clean symbol (remove suffix like .sa)
        clean_symbol = symbol.split('.')[0].upper()
        
        return any(clean_symbol == s.upper() for s in group_symbols)
    
    def _get_symbol_group(self, symbol: str) -> str:
        """Get group name for a symbol."""
        clean_symbol = symbol.split('.')[0].upper()
        groups = self.config.get('groups', {})
        
        for group_name, symbols in groups.items():
            if any(clean_symbol == s.upper() for s in symbols):
                return group_name
        
        return "unknown"


class ProfitLock:
    """
    Profit lock / trailing stop manager.
    Implements profit-locking ladder.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Default profit lock ladder (pips profit -> lock at pips)
        self.ladder = self.config.get('profit_ladder', [
            {"trigger": 10, "lock": 0},      # +10 pips -> move SL to breakeven
            {"trigger": 20, "lock": 10},     # +20 pips -> lock 10 pips profit
            {"trigger": 40, "lock": 25},     # +40 pips -> lock 25 pips profit
            {"trigger": 60, "lock": 45},     # +60 pips -> lock 45 pips profit
            {"trigger": 100, "lock": 80},    # +100 pips -> lock 80 pips profit
        ])
        self.ladder_r = self.config.get('profit_ladder_r', [])
        self.be_plus_cfg = self.config.get('be_plus', {})
        self.be_plus_enabled = self.be_plus_cfg.get('enabled', True)
        self.be_plus_trigger_r = float(self.be_plus_cfg.get('trigger_r', 0.30))
        self.be_plus_lock_r = float(self.be_plus_cfg.get('lock_r', 0.10))
        self.disallow_breakeven = bool(self.be_plus_cfg.get('disallow_breakeven', True))
        self.min_r_before_lock = float(self.be_plus_cfg.get('min_r_before_lock', 0.0))
    
    def get_new_sl(
        self,
        position: Dict[str, Any],
        current_price: float,
        pip_value: float = 0.0001
    ) -> Optional[float]:
        """
        Calculate new SL based on profit lock rules.
        
        Args:
            position: Position dictionary
            current_price: Current market price
            pip_value: Value of one pip for this symbol
        
        Returns:
            New SL price if lock should be applied, None otherwise
        """
        entry = position.get('price_open', 0)
        current_sl = position.get('sl', 0)
        direction = position.get('type', 'BUY')
        
        if entry <= 0 or pip_value <= 0:
            return None
        
        # Calculate current profit in pips
        if direction == 'BUY':
            profit_pips = (current_price - entry) / pip_value
        else:
            profit_pips = (entry - current_price) / pip_value

        # R-based BE+ buffer to avoid zero-breakeven after positive profit
        be_plus_lock_pips = None
        risk_pips = None
        if self.be_plus_enabled:
            initial_sl = (
                position.get('sl_initial')
                or position.get('sl_open')
                or position.get('sl')
            )
            if initial_sl:
                if direction == 'BUY' and initial_sl < entry:
                    risk_pips = abs(entry - initial_sl) / pip_value
                elif direction != 'BUY' and initial_sl > entry:
                    risk_pips = abs(entry - initial_sl) / pip_value

            if risk_pips and risk_pips > 0:
                trigger_pips = self.be_plus_trigger_r * risk_pips
                if profit_pips >= trigger_pips:
                    be_plus_lock_pips = self.be_plus_lock_r * risk_pips
        else:
            initial_sl = (
                position.get('sl_initial')
                or position.get('sl_open')
                or position.get('sl')
            )
            if initial_sl:
                if direction == 'BUY' and initial_sl < entry:
                    risk_pips = abs(entry - initial_sl) / pip_value
                elif direction != 'BUY' and initial_sl > entry:
                    risk_pips = abs(entry - initial_sl) / pip_value

        if risk_pips and risk_pips > 0 and self.min_r_before_lock > 0:
            if profit_pips < (self.min_r_before_lock * risk_pips):
                return None
        
        # Find highest triggered lock level (R-based ladder takes precedence if available)
        new_lock_pips = None
        if self.ladder_r and risk_pips and risk_pips > 0:
            profit_r = profit_pips / risk_pips
            for level in sorted(self.ladder_r, key=lambda x: x['trigger_r'], reverse=True):
                if profit_r >= level['trigger_r']:
                    new_lock_pips = level['lock_r'] * risk_pips
                    break
        else:
            for level in sorted(self.ladder, key=lambda x: x['trigger'], reverse=True):
                if profit_pips >= level['trigger']:
                    new_lock_pips = level['lock']
                    break
        
        if new_lock_pips is None and be_plus_lock_pips is None:
            return None  # No lock triggered

        # Disallow breakeven locks once trade is positive
        if self.disallow_breakeven and new_lock_pips <= 0:
            new_lock_pips = None

        if be_plus_lock_pips is not None:
            if new_lock_pips is None or be_plus_lock_pips > new_lock_pips:
                new_lock_pips = be_plus_lock_pips

        if new_lock_pips is None:
            return None
        
        # Calculate new SL price
        if direction == 'BUY':
            new_sl = entry + (new_lock_pips * pip_value)
            # Only move SL if it's better (higher)
            if current_sl > 0 and new_sl <= current_sl:
                return None
        else:
            new_sl = entry - (new_lock_pips * pip_value)
            # Only move SL if it's better (lower)
            if current_sl > 0 and new_sl >= current_sl:
                return None
        
        return new_sl


def create_risk_manager(config: Dict[str, Any] = None) -> RiskManager:
    """Factory function to create a RiskManager."""
    return RiskManager(config)


def create_profit_lock(config: Dict[str, Any] = None) -> ProfitLock:
    """Factory function to create a ProfitLock."""
    return ProfitLock(config)


