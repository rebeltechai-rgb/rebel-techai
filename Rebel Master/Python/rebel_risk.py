"""
REBEL Risk - Risk Management Module
Enforces 3 core rules:
  1. 3% risk per trade (proper volume calculation)
  2. 30% max equity exposure (margin cap)
  3. 1 open position per symbol
"""

import MetaTrader5 as mt5
from typing import Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from connectors.base_connector import BrokerConnector


# =============================================================================
# OIL SYMBOL OVERRIDES - Fix broker's incorrect tick/contract data
# =============================================================================
OIL_CONTRACT_OVERRIDES = {
    "USOIL": 1000,
    "UKOIL": 1000,
    "WTI": 1000,
    "BRENT": 1000,
    "WTI.FS": 1000,
    "BRENT.FS": 1000,
    "USOIL.A": 1000,
    "UKOIL.A": 1000,
}

OIL_MAX_VOLUME = {
    "USOIL": 0.5,
    "UKOIL": 0.5,
    "WTI": 0.5,
    "BRENT": 0.5,
    "WTI.FS": 0.5,
    "BRENT.FS": 0.5,
    "USOIL.A": 0.5,
    "UKOIL.A": 0.5,
}


# =============================================================================
# STANDALONE FUNCTION: 3% Risk Volume Calculator
# =============================================================================
def calculate_volume_3pct(
    symbol: str,
    entry_price: float,
    sl_price: float,
    equity: float,
    risk_pct: float = 0.03,
) -> float:
    """
    Calculate volume so that max loss (at SL) = equity * 3%.
    
    Uses MT5's tick_value, tick_size, and trade_contract_size for accurate calculation.
    Includes special overrides for OIL symbols where broker data is often incorrect.
    
    Args:
        symbol: Trading symbol
        entry_price: Entry price
        sl_price: Stop loss price
        equity: Account equity
        risk_pct: Risk percentage (default 0.03 = 3%)
    
    Returns:
        Calculated volume (lot size)
    """
    sym = symbol.upper()
    
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"[LOT] {symbol}: Cannot get symbol info")
        return 0.0
    
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    contract_size = getattr(symbol_info, "trade_contract_size", 1.0)
    
    # ---- OIL OVERRIDES (for brokers with bad tick data) ----
    if sym in OIL_CONTRACT_OVERRIDES:
        # If broker gives weirdly tiny tick values, override
        if tick_value <= 0 or tick_value < 0.01:
            tick_value = 10.0
            print(f"[LOT] {symbol}: Using OIL override tick_value={tick_value}")
        if tick_size <= 0:
            tick_size = 0.01
            print(f"[LOT] {symbol}: Using OIL override tick_size={tick_size}")
    # ----------------------------------
    
    # Distance from entry to SL
    sl_dist = abs(entry_price - sl_price)
    if sl_dist <= 0:
        print(f"[LOT] {symbol}: SL distance is zero or negative, cannot size position.")
        return 0.0
    
    # Basic tick safety
    if tick_size is None or tick_size <= 0:
        print(f"[LOT] {symbol}: invalid tick_size={tick_size}, cannot size position.")
        return 0.0
    if tick_value is None or tick_value <= 0:
        print(f"[LOT] {symbol}: invalid tick_value={tick_value}, cannot size position.")
        return 0.0
    
    ticks = sl_dist / tick_size
    
    # ⭐ IMPORTANT FIX:
    # In MT5, tick_value already includes the contract size for 1 lot.
    # So the correct loss per lot is:
    #   loss_per_lot = ticks * tick_value
    # Multiplying by contract_size again overstates risk by 10x–100x.
    loss_per_lot = ticks * tick_value
    
    if loss_per_lot <= 0:
        print(f"[LOT] {symbol}: non-positive loss_per_lot={loss_per_lot}, cannot size position.")
        return 0.0
    
    risk_amount = equity * risk_pct
    volume = risk_amount / loss_per_lot
    
    # Log the calculation
    print(f"[LOT] {symbol}: Equity=${equity:.2f}, Risk={risk_pct*100:.0f}% (${risk_amount:.2f})")
    print(f"[LOT] {symbol}: tick_value={tick_value}, tick_size={tick_size}, contract={contract_size}")
    print(f"[LOT] {symbol}: SL_dist={sl_dist:.5f}, ticks={ticks:.0f}, loss/lot=${loss_per_lot:.2f}")
    
    # Broker min/max/step
    vol_min = symbol_info.volume_min
    vol_step = symbol_info.volume_step or 0.01
    vol_max = symbol_info.volume_max
    
    # Extra per-symbol cap for oil as last safety layer
    max_vol_override = OIL_MAX_VOLUME.get(sym)
    if max_vol_override is not None:
        vol_max = min(vol_max, max_vol_override)
        print(f"[LOT] {symbol}: OIL max volume cap = {max_vol_override}")
    
    # Clamp & step
    volume = max(vol_min, min(volume, vol_max))
    steps = round(volume / vol_step)
    volume = steps * vol_step
    
    if volume < vol_min:
        print(f"[LOT] {symbol}: Volume too small, returning 0")
        return 0.0
    
    # General safety cap
    max_safe_lot = 1.0
    if volume > max_safe_lot:
        print(f"[LOT] {symbol}: CAPPED from {volume:.2f} to {max_safe_lot} (safety cap)")
        volume = max_safe_lot
    
    actual_risk = volume * loss_per_lot
    print(f"[LOT] {symbol}: Final={volume:.2f} lots (risks ${actual_risk:.2f} = {actual_risk/equity*100:.1f}%)")
    
    return volume


class RebelRisk:
    """
    Risk management enforcing:
    - 3% risk per trade via tick_value/tick_size/contract_size
    - 30% equity exposure cap
    - 1 position per symbol
    """
    
    def __init__(self, broker: "BrokerConnector" = None, config: dict = None):
        self.broker = broker
        self.config = config or {}
        
        # Load from risk_engine config section
        risk_cfg = self.config.get("risk_engine", {})
        
        # RULE 1: 3% risk per trade
        self.risk_per_trade_pct = risk_cfg.get("percent_risk_per_trade", 3.0) / 100.0
        
        # RULE 2: 30% max equity exposure
        self.max_equity_exposure_pct = risk_cfg.get("max_equity_exposure_pct", 30.0) / 100.0
        
        # Legacy settings (for backward compatibility)
        self.max_open_trades = self.config.get("risk", {}).get("max_open_trades", 15)
        self.max_daily_drawdown = self.config.get("risk", {}).get("max_daily_drawdown", 0.15)
        self.max_risk_per_trade = self.risk_per_trade_pct  # Alias for legacy code
        self.min_free_margin_pct = 1.0 - self.max_equity_exposure_pct  # 70% free if 30% max used
    
    def set_broker(self, broker: "BrokerConnector"):
        """Set the broker connector."""
        self.broker = broker
    
    # =========================================================================
    # RULE 1: Calculate lot size to risk exactly 3% using MT5 tick data
    # =========================================================================
    def calc_lot_for_risk(self, symbol: str, entry_price: float, sl_price: float) -> Tuple[float, str]:
        """
        Calculate lot size to risk exactly 3% of equity.
        
        Uses MT5's tick_value, tick_size, and trade_contract_size for accurate calculation.
        
        Formula: lot = risk_amount / (sl_ticks * tick_value)
        
        Returns:
            Tuple of (lot_size, message)
        """
        # Get account info
        account = mt5.account_info()
        if account is None:
            return 0.01, "Failed to get account info, using min lot"
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.01, f"Symbol {symbol} not found, using min lot"
        
        # Calculate risk amount (3% of equity)
        equity = account.equity
        risk_amount = equity * self.risk_per_trade_pct
        
        # Get MT5 symbol specs
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        contract_size = symbol_info.trade_contract_size
        vol_min = symbol_info.volume_min
        vol_max = symbol_info.volume_max
        vol_step = symbol_info.volume_step or 0.01
        
        # Validate
        if tick_size <= 0 or tick_value <= 0:
            return vol_min, f"Invalid tick data for {symbol}"
        
        # Calculate SL distance
        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            return vol_min, "Invalid SL distance"
        
        # Convert to ticks
        sl_ticks = sl_distance / tick_size
        
        # Money lost per 1.0 lot if SL hit
        loss_per_lot = sl_ticks * tick_value
        
        if loss_per_lot <= 0:
            return vol_min, "Invalid loss calculation"
        
        # Calculate lot: risk_amount / loss_per_lot
        lot = risk_amount / loss_per_lot
        
        # Normalize to broker constraints
        lot = round(lot / vol_step) * vol_step
        lot = max(vol_min, min(lot, vol_max))
        
        actual_risk = lot * loss_per_lot
        msg = f"Lot={lot:.2f} risks ${actual_risk:.2f} ({self.risk_per_trade_pct*100:.0f}% of ${equity:.2f})"
        
        return lot, msg
    
    # =========================================================================
    # RULE 2: Check 30% equity exposure cap
    # =========================================================================
    def check_equity_exposure(self) -> Tuple[bool, float, str]:
        """
        Check if current margin usage is under 30% of equity.
        
        Returns:
            Tuple of (can_trade, current_exposure_pct, message)
        """
        account = mt5.account_info()
        if account is None:
            return False, 0.0, "Cannot get account info"
        
        equity = account.equity
        margin_used = account.margin
        
        if equity <= 0:
            return False, 0.0, "No equity"
        
        current_exposure = margin_used / equity
        
        if current_exposure >= self.max_equity_exposure_pct:
            return False, current_exposure, f"BLOCKED: {current_exposure*100:.1f}% >= {self.max_equity_exposure_pct*100:.0f}% limit"
        
        return True, current_exposure, f"OK: {current_exposure*100:.1f}% < {self.max_equity_exposure_pct*100:.0f}%"
    
    def check_trade_would_exceed_exposure(self, symbol: str, volume: float, price: float, direction: str) -> Tuple[bool, str]:
        """
        Check if opening a new trade would exceed 30% equity exposure.
        
        Returns:
            Tuple of (would_exceed, message)
        """
        account = mt5.account_info()
        if account is None:
            return True, "Cannot get account info"
        
        equity = account.equity
        margin_used = account.margin
        
        # Calculate margin for new trade
        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
        new_margin = mt5.order_calc_margin(order_type, symbol, volume, price)
        
        if new_margin is None:
            return True, "Cannot calculate margin"
        
        projected_margin = margin_used + new_margin
        projected_exposure = projected_margin / equity if equity > 0 else 1.0
        
        if projected_exposure > self.max_equity_exposure_pct:
            return True, f"Would be {projected_exposure*100:.1f}% > {self.max_equity_exposure_pct*100:.0f}% limit"
        
        return False, f"Would be {projected_exposure*100:.1f}% (OK)"
    
    # =========================================================================
    # RULE 3: Only 1 position per symbol
    # =========================================================================
    def has_open_position(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if symbol already has an open position.
        
        Returns:
            Tuple of (has_position, message)
        """
        positions = mt5.positions_get(symbol=symbol)
        
        if positions and len(positions) > 0:
            return True, f"{symbol} already has {len(positions)} position(s)"
        
        return False, f"No open position on {symbol}"
    
    # =========================================================================
    # Combined validation for all 3 rules
    # =========================================================================
    def validate_new_trade(self, symbol: str, entry_price: float, sl_price: float, direction: str) -> Dict:
        """
        Validate a potential trade against all 3 rules.
        
        Returns:
            Dict with:
                - can_trade: bool
                - lot: calculated lot size
                - messages: list of check results
        """
        result = {
            "can_trade": True,
            "lot": 0.01,
            "messages": []
        }
        
        # RULE 3: Check 1 position per symbol first (cheapest check)
        has_pos, msg = self.has_open_position(symbol)
        result["messages"].append(f"[RULE 3] {msg}")
        if has_pos:
            result["can_trade"] = False
            return result
        
        # RULE 2: Check 30% equity exposure
        can_trade, exposure, msg = self.check_equity_exposure()
        result["messages"].append(f"[RULE 2] {msg}")
        if not can_trade:
            result["can_trade"] = False
            return result
        
        # RULE 1: Calculate lot for 3% risk
        lot, msg = self.calc_lot_for_risk(symbol, entry_price, sl_price)
        result["lot"] = lot
        result["messages"].append(f"[RULE 1] {msg}")
        
        # Final check: would this trade exceed 30%?
        would_exceed, msg = self.check_trade_would_exceed_exposure(symbol, lot, entry_price, direction)
        result["messages"].append(f"[MARGIN] {msg}")
        if would_exceed:
            result["can_trade"] = False
        
        return result
    
    def get_open_positions_count(self) -> int:
        """Get number of currently open positions."""
        if self.broker is None:
            return 0
        return self.broker.get_positions_count()
    
    def get_open_positions(self, symbol: str = None) -> list:
        """
        Get list of open positions.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of position dictionaries
        """
        if self.broker is None:
            return []
        return self.broker.get_positions(symbol)
    
    def check_max_trades(self) -> tuple:
        """
        Check if maximum open trades limit is reached.
        
        Returns:
            Tuple of (can_trade: bool, message: str)
        """
        current_positions = self.get_open_positions_count()
        
        if current_positions >= self.max_open_trades:
            return False, f"Max trades reached ({current_positions}/{self.max_open_trades})"
        
        return True, f"Open positions: {current_positions}/{self.max_open_trades}"
    
    def check_free_margin(self) -> tuple:
        """
        Check if there is sufficient free margin.
        
        Returns:
            Tuple of (can_trade: bool, message: str)
        """
        if self.broker is None:
            return False, "No broker connected"
        
        account = self.broker.get_account_info()
        if account is None:
            return False, "Unable to get account info"
        
        equity = account.get("equity", 0)
        free_margin = account.get("free_margin", 0)
        
        if equity <= 0:
            return False, "No equity available"
        
        free_margin_pct = free_margin / equity
        
        if free_margin_pct < self.min_free_margin_pct:
            return False, f"Insufficient free margin: {free_margin_pct:.1%}"
        
        return True, f"Free margin: {free_margin_pct:.1%}"
    
    def check_daily_drawdown(self) -> tuple:
        """
        Check if daily drawdown limit is exceeded.
        
        Returns:
            Tuple of (can_trade: bool, message: str)
        """
        if self.broker is None:
            return False, "No broker connected"
        
        account = self.broker.get_account_info()
        if account is None:
            return False, "Unable to get account info"
        
        balance = account.get("balance", 0)
        equity = account.get("equity", 0)
        
        # Simple drawdown check: compare equity to balance
        if balance <= 0:
            return False, "Invalid balance"
        
        current_drawdown = (balance - equity) / balance
        
        if current_drawdown >= self.max_daily_drawdown:
            return False, f"Daily drawdown limit reached: {current_drawdown:.1%}"
        
        return True, f"Current drawdown: {current_drawdown:.1%}"
    
    def check_symbol_exposure(self, symbol: str) -> tuple:
        """
        Check if there's already an open position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (can_trade: bool, message: str)
        """
        positions = self.get_open_positions(symbol)
        
        if len(positions) > 0:
            return False, f"Already have position on {symbol}"
        
        return True, f"No existing position on {symbol}"
    
    def calculate_lot_size(self, symbol: str, stop_loss_pips: float) -> Optional[float]:
        """
        Calculate position size based on risk percentage.
        
        Args:
            symbol: Trading symbol
            stop_loss_pips: Stop loss distance in pips
            
        Returns:
            Lot size or None if calculation fails
        """
        if self.broker is None:
            return None
        
        account = self.broker.get_account_info()
        if account is None:
            return None
        
        symbol_info = self.broker.get_symbol_info(symbol)
        if symbol_info is None:
            return None
        
        balance = account.get("balance", 0)
        
        # Risk amount in account currency
        risk_amount = balance * self.max_risk_per_trade
        
        # Get pip value (simplified calculation)
        tick_size = symbol_info.get("trade_tick_size", 0)
        tick_value = symbol_info.get("trade_tick_value", 0)
        volume_min = symbol_info.get("volume_min", 0.01)
        volume_max = symbol_info.get("volume_max", 100)
        volume_step = symbol_info.get("volume_step", 0.01)
        
        if tick_size == 0 or stop_loss_pips == 0:
            return volume_min
        
        # Calculate pip value per lot
        pip_value = tick_value / tick_size * 0.0001  # Assuming 4/5 decimal broker
        
        if pip_value <= 0:
            return volume_min
        
        # Calculate lot size
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Round to symbol's lot step
        if volume_step > 0:
            lot_size = round(lot_size / volume_step) * volume_step
        
        # Enforce min/max limits
        lot_size = max(lot_size, volume_min)
        lot_size = min(lot_size, volume_max)
        
        return round(lot_size, 2)
    
    def validate_trade(self, symbol: str) -> dict:
        """
        Run all risk checks for a potential trade.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "can_trade": True,
            "checks": {},
            "messages": []
        }
        
        # Run all checks
        checks = [
            ("max_trades", self.check_max_trades()),
            ("free_margin", self.check_free_margin()),
            ("daily_drawdown", self.check_daily_drawdown()),
            ("symbol_exposure", self.check_symbol_exposure(symbol)),
        ]
        
        for check_name, (passed, message) in checks:
            result["checks"][check_name] = passed
            result["messages"].append(f"{check_name}: {message}")
            
            if not passed:
                result["can_trade"] = False
        
        return result
    
    def get_risk_summary(self) -> dict:
        """
        Get current risk metrics summary.
        
        Returns:
            Dictionary with risk metrics
        """
        if self.broker is None:
            return {}
        
        account = self.broker.get_account_info()
        if account is None:
            return {}
        
        positions = self.get_open_positions_count()
        
        return {
            "balance": account.get("balance", 0),
            "equity": account.get("equity", 0),
            "margin_used": account.get("margin", 0),
            "free_margin": account.get("free_margin", 0),
            "profit": account.get("profit", 0),
            "open_positions": positions,
            "max_positions": self.max_open_trades,
            "risk_per_trade": f"{self.max_risk_per_trade:.1%}",
            "max_drawdown": f"{self.max_daily_drawdown:.1%}"
        }
