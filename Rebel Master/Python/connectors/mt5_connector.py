"""
REBEL MT5 Connector - MetaTrader 5 Broker Implementation
Implements the BrokerConnector interface for MetaTrader 5.
"""

import MetaTrader5 as mt5
import pandas as pd
from typing import Optional, Dict, Any
from .base_connector import BrokerConnector

# Default MT5 terminal path (can be overridden by config)
TERMINAL_PATH = r"C:\Master Pro\terminal64.exe"

# =============================================================================
# SPECIAL FILLING MODES - Override for problematic symbols
# 10030 = INVALID_FILL error means broker doesn't support that fill mode
# =============================================================================
SPECIAL_FILLING = {
    # Softs/Futures - require FOK
    "SOYBEAN.FS": mt5.ORDER_FILLING_FOK,
    "CORN.FS": mt5.ORDER_FILLING_FOK,
    "WHEAT.FS": mt5.ORDER_FILLING_FOK,
    "COFFEE.FS": mt5.ORDER_FILLING_FOK,
    "COCOA.FS": mt5.ORDER_FILLING_FOK,
    "SUGAR.FS": mt5.ORDER_FILLING_FOK,
    "NGAS.FS": mt5.ORDER_FILLING_FOK,
    "WTI.FS": mt5.ORDER_FILLING_FOK,
    "BRENT.FS": mt5.ORDER_FILLING_FOK,
    
    # Crypto USD - require IOC (Axi available symbols)
    "BTCUSD": mt5.ORDER_FILLING_IOC,
    "ETHUSD": mt5.ORDER_FILLING_IOC,
    "LTCUSD": mt5.ORDER_FILLING_IOC,
    "BCHUSD": mt5.ORDER_FILLING_IOC,
    "XRPUSD": mt5.ORDER_FILLING_IOC,
    "ADAUSD": mt5.ORDER_FILLING_IOC,
    "DOGUSD": mt5.ORDER_FILLING_IOC,
    "DOTUSD": mt5.ORDER_FILLING_IOC,
    "XLMUSD": mt5.ORDER_FILLING_IOC,
    "LNKUSD": mt5.ORDER_FILLING_IOC,
    # Crypto hyphenated (Axi available)
    "SOL-USD": mt5.ORDER_FILLING_IOC,
    "AVAX-USD": mt5.ORDER_FILLING_IOC,
    "AAVE-USD": mt5.ORDER_FILLING_IOC,
    "UNI-USD": mt5.ORDER_FILLING_IOC,
    "SUSHI-USD": mt5.ORDER_FILLING_IOC,
    "COMP-USD": mt5.ORDER_FILLING_IOC,
    "CRV-USD": mt5.ORDER_FILLING_IOC,
    "LRC-USD": mt5.ORDER_FILLING_IOC,
    "MANA-USD": mt5.ORDER_FILLING_IOC,
    "SAND-USD": mt5.ORDER_FILLING_IOC,
    "BAT-USD": mt5.ORDER_FILLING_IOC,
    "BNB-USD": mt5.ORDER_FILLING_IOC,
    "KSM-USD": mt5.ORDER_FILLING_IOC,
    "XTZ-USD": mt5.ORDER_FILLING_IOC,
    # Crypto JPY pairs (Axi available)
    "BCH-JPY": mt5.ORDER_FILLING_IOC,
    "BTC-JPY": mt5.ORDER_FILLING_IOC,
    "ETH-JPY": mt5.ORDER_FILLING_IOC,
    "LNK-JPY": mt5.ORDER_FILLING_IOC,
    "LTC-JPY": mt5.ORDER_FILLING_IOC,
    "XRP-JPY": mt5.ORDER_FILLING_IOC,
    
    # VIX - requires IOC (FOK rejected by Axi)
    "VIX.FS": mt5.ORDER_FILLING_IOC,
    
    # Additional futures/indices (IOC required - FOK rejected by Axi)
    "NATGAS.FS": mt5.ORDER_FILLING_IOC,
    "COPPER.FS": mt5.ORDER_FILLING_IOC,
    "CAC40.FS": mt5.ORDER_FILLING_IOC,
    "EUSTX50.FS": mt5.ORDER_FILLING_IOC,
    "NK225.FS": mt5.ORDER_FILLING_IOC,
    "S&P.FS": mt5.ORDER_FILLING_IOC,
    "SPI200.FS": mt5.ORDER_FILLING_IOC,
    "DJ30.FS": mt5.ORDER_FILLING_IOC,
    "FT100.FS": mt5.ORDER_FILLING_IOC,
    "USDINDEX.FS": mt5.ORDER_FILLING_IOC,
    
    # Indices - require IOC
    "US500": mt5.ORDER_FILLING_IOC,
    "US500.FS": mt5.ORDER_FILLING_IOC,
    "US30": mt5.ORDER_FILLING_IOC,
    "US30.FS": mt5.ORDER_FILLING_IOC,
    "USTECH": mt5.ORDER_FILLING_IOC,
    "USTECH.FS": mt5.ORDER_FILLING_IOC,
    "NAS100": mt5.ORDER_FILLING_IOC,
    "NAS100.FS": mt5.ORDER_FILLING_IOC,
    "US2000": mt5.ORDER_FILLING_IOC,
    "US2000.FS": mt5.ORDER_FILLING_IOC,
    "UK100": mt5.ORDER_FILLING_IOC,
    "UK100.FS": mt5.ORDER_FILLING_IOC,
    "GER40": mt5.ORDER_FILLING_IOC,
    "GER40.FS": mt5.ORDER_FILLING_IOC,
    "DAX40": mt5.ORDER_FILLING_IOC,
    "DAX40.FS": mt5.ORDER_FILLING_IOC,
    "AUS200": mt5.ORDER_FILLING_IOC,
    "AUS200.FS": mt5.ORDER_FILLING_IOC,
    "JPN225": mt5.ORDER_FILLING_IOC,
    "JPN225.FS": mt5.ORDER_FILLING_IOC,
    "HK50": mt5.ORDER_FILLING_IOC,
    "HK50.FS": mt5.ORDER_FILLING_IOC,
    "HSI.FS": mt5.ORDER_FILLING_IOC,
    "IT40": mt5.ORDER_FILLING_IOC,
    "SWI20": mt5.ORDER_FILLING_IOC,
    "CHINA50": mt5.ORDER_FILLING_IOC,
    "CHINA50.FS": mt5.ORDER_FILLING_IOC,
    "EU50": mt5.ORDER_FILLING_IOC,
    "EU50.FS": mt5.ORDER_FILLING_IOC,
    "FRA40": mt5.ORDER_FILLING_IOC,
    "FRA40.FS": mt5.ORDER_FILLING_IOC,
    
    # Metals - require IOC
    "XAUUSD": mt5.ORDER_FILLING_IOC,
    "XAGUSD": mt5.ORDER_FILLING_IOC,
    
    # Energies - require IOC
    "UKOIL": mt5.ORDER_FILLING_IOC,
    "USOIL": mt5.ORDER_FILLING_IOC,
    # Add more as needed
}

# Symbols that reject explicit filling modes; let broker choose
NO_FILLING_MODE = {
    "CHINA50",
    "CHINA50.FS",
}


class MT5Connector(BrokerConnector):
    """MetaTrader 5 broker connector implementation."""
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    
    def __init__(self, config: dict = None):
        """
        Initialize the MT5 connector.
        
        Args:
            config: Configuration dictionary with optional MT5 settings
        """
        self.config = config or {}
        self.mt5_config = self.config.get("mt5", {})
        self.connected = False
    
    def initialize(self, max_retries: int = 5, retry_delay: int = 3) -> bool:
        """
        Initialize connection to MetaTrader 5 with retry logic.
        
        Args:
            max_retries: Number of connection attempts
            retry_delay: Seconds to wait between retries
            
        Returns:
            True if connection was successful
        """
        import time
        
        # Use config path, or fall back to hard-coded TERMINAL_PATH
        path = self.mt5_config.get("path") or TERMINAL_PATH
        
        for attempt in range(1, max_retries + 1):
            print(f"[MT5] Connection attempt {attempt}/{max_retries}...")
            
            # Shutdown any existing connection first
            try:
                mt5.shutdown()
            except:
                pass
            
            # Small delay to let MT5 settle
            time.sleep(1)
            
            # Initialize MT5 with explicit path
            print(f"[MT5] Initializing with terminal: {path}")
            if not mt5.initialize(path=path):
                error = mt5.last_error()
                print(f"[MT5] Initialization failed: {error}")
                if attempt < max_retries:
                    print(f"[MT5] Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                continue
            
            # Wait for terminal to fully load (handles "resetting platform" state)
            time.sleep(2)
            
            # Verify terminal info with retries
            terminal_ready = False
            for check in range(3):
                terminal_info = mt5.terminal_info()
                if terminal_info is not None:
                    terminal_ready = True
                    break
                print(f"[MT5] Waiting for terminal info... ({check+1}/3)")
                time.sleep(2)
            
            if not terminal_ready:
                print("[MT5] Failed to get terminal info - platform may be resetting")
                if attempt < max_retries:
                    print(f"[MT5] Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                continue
            
            # Verify account info
            account_info = mt5.account_info()
            if account_info is None:
                print("[MT5] Failed to get account info - check if logged in")
                if attempt < max_retries:
                    print(f"[MT5] Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                continue
            
            # Success!
            print(f"[MT5] Connected successfully on attempt {attempt}")
            print(f"[MT5] Account: {account_info.login} | Balance: {account_info.balance}")
            print(f"[MT5] Server: {account_info.server}")
            
            self.connected = True
            return True
        
        # All retries exhausted
        print(f"[MT5] Failed to connect after {max_retries} attempts")
        self.connected = False
        return False
    
    def shutdown(self) -> None:
        """Shutdown the MT5 connection."""
        mt5.shutdown()
        self.connected = False
        print("[MT5] Disconnected")
    
    def get_historical_data(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from MT5.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., 'M5', 'M15', 'H1')
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data or None
        """
        mt5_tf = self.TIMEFRAME_MAP.get(timeframe)
        if mt5_tf is None:
            print(f"[MT5] Unknown timeframe: {timeframe}")
            return None
        
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
        
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        
        return df
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current bid/ask prices.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with 'bid' and 'ask' keys
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "time": tick.time
        }
    
    def get_spread(self, symbol: str) -> Optional[float]:
        """
        Get current spread for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Spread in price terms
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return abs(tick.ask - tick.bid)
    
    def place_market_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "",
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Place a market order via MT5.
        
        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            volume: Lot size
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            extra: Additional parameters (magic, deviation, etc.)
            
        Returns:
            Dict with order result
        """
        extra = extra or {}
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {
                "ok": False,
                "retcode": -1,
                "order": None,
                "message": "Failed to get tick data"
            }
        
        # Determine order type and price
        if direction == "long":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        # Get symbol info for rounding
        info = mt5.symbol_info(symbol)
        if info is not None:
            digits = info.digits
            price = round(price, digits)
            if sl is not None:
                sl = round(sl, digits)
            if tp is not None:
                tp = round(tp, digits)
        
        # Get correct filling mode for this symbol (auto-detected)
        filling_mode = self.get_correct_filling_mode(symbol)
        
        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": extra.get("deviation", 10),
            "magic": extra.get("magic", 20251202),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        
        # Add SL/TP if provided
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        sym_upper = symbol.upper()
        allow_fallback = True
        if sym_upper in NO_FILLING_MODE:
            request.pop("type_filling", None)
            allow_fallback = False
            print(f"[MT5] {symbol}: forcing broker default filling mode")

        # Send order (with filling-mode fallback)
        result, used_mode = self._send_order_with_fill_fallback(request, allow_fallback=allow_fallback)
        
        if result is None:
            return {
                "ok": False,
                "retcode": -1,
                "order": None,
                "message": "order_send returned None"
            }
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            mode_label = "AUTO" if used_mode is None else used_mode
            print(f"[MT5] {symbol} order success (filling_mode={mode_label})")
            return {
                "ok": True,
                "retcode": result.retcode,
                "order": result.order,
                "message": result.comment if hasattr(result, 'comment') else "",
                "price": price,
                "volume": volume,
                "filling_mode_used": used_mode if used_mode is not None else filling_mode
            }
        
        print(f"[MT5] {symbol} order failed: {result.retcode} - {result.comment}")
        return {
            "ok": False,
            "retcode": result.retcode,
            "order": None,
            "message": result.comment if hasattr(result, 'comment') else "",
            "price": price,
            "volume": volume
        }
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current account information from MT5.
        
        Returns:
            Dict with account info
        """
        info = mt5.account_info()
        if info is None:
            return None
        
        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "leverage": info.leverage,
            "profit": info.profit,
            "server": info.server,
            "currency": info.currency
        }
    
    def ensure_symbol(self, symbol: str) -> bool:
        """
        Ensure a symbol is available and visible in Market Watch.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if symbol is available
        """
        # Ensure MT5 is initialized (can drop between scans)
        if mt5.terminal_info() is None:
            if not self.initialize():
                return False

        info = mt5.symbol_info(symbol)
        if info is None:
            return False
        
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                return False
        
        return True
    
    def get_correct_filling_mode(self, symbol: str) -> int:
        """
        Auto-detect the correct filling mode for a symbol.
        
        Checks:
        1. Special filling table (hardcoded overrides)
        2. Broker-specified required mode from symbol info
        3. Exchange execution mode (futures/softs use FOK)
        4. Default fallback to IOC (most CFD brokers prefer this)
        
        Args:
            symbol: Trading symbol
            
        Returns:
            MT5 filling mode constant
        """
        sym_upper = symbol.upper()
        
        # 1. Check special filling table first (exact match)
        if sym_upper in SPECIAL_FILLING:
            mode = SPECIAL_FILLING[sym_upper]
            mode_name = {0: "FOK", 1: "IOC", 2: "RETURN"}.get(mode, str(mode))
            print(f"[MT5] {symbol}: Using special filling mode ({mode_name})")
            return mode
        
        # 2. Get symbol info
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC  # Default for unknown symbols
        
        # 3. Prefer broker-provided filling mode if it's a single allowed value
        if hasattr(info, 'filling_mode'):
            fm = info.filling_mode
            if fm in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
                return fm
            # Some brokers report a bitmask: 1=FOK, 2=IOC, 4=RETURN
            if isinstance(fm, int):
                if fm & 2:  # IOC supported
                    return mt5.ORDER_FILLING_IOC
                if fm & 1:  # FOK supported
                    return mt5.ORDER_FILLING_FOK
                if fm & 4:  # RETURN supported
                    return mt5.ORDER_FILLING_RETURN
        
        # 4. Futures and soft commodities often use FOK only
        if hasattr(info, 'execution_mode') and info.execution_mode == mt5.SYMBOL_TRADE_EXECUTION_EXCHANGE:
            return mt5.ORDER_FILLING_FOK
        
        # 5. Default fallback to IOC (most CFD brokers like Axi prefer this)
        return mt5.ORDER_FILLING_IOC

    def _send_order_with_fill_fallback(
        self,
        request: Dict[str, Any],
        allow_fallback: bool = True
    ) -> tuple[Any, Optional[int]]:
        """Send order with optional filling-mode fallbacks; returns (result, mode_used)."""
        preferred = request.get("type_filling")
        if not allow_fallback:
            candidates = [preferred]
        else:
            candidates = [preferred, mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, None]
        seen = set()
        ordered = []
        for mode in candidates:
            key = mode if mode is not None else "NONE"
            if key in seen:
                continue
            seen.add(key)
            ordered.append(mode)

        invalid_fill_code = getattr(mt5, "TRADE_RETCODE_INVALID_FILL", None)
        last_result = None
        last_mode = None
        base_request = dict(request)

        for mode in ordered:
            req = dict(base_request)
            if mode is None:
                req.pop("type_filling", None)
            else:
                req["type_filling"] = mode

            result = mt5.order_send(req)
            last_result = result
            last_mode = mode
            if result is None:
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return result, mode

            is_invalid_fill = (
                (invalid_fill_code is not None and result.retcode == invalid_fill_code)
                or ("Unsupported filling mode" in str(result.comment))
            )
            if is_invalid_fill:
                continue
            return result, mode

        return last_result, last_mode
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed symbol information from MT5.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with symbol info
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        
        return {
            "point": info.point,
            "digits": info.digits,
            "trade_tick_value": info.trade_tick_value,
            "trade_tick_size": info.trade_tick_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_contract_size": info.trade_contract_size,
            "spread": info.spread,
            "visible": info.visible
        }
    
    def get_positions_count(self) -> int:
        """
        Get the number of open positions.
        
        Returns:
            Number of open positions
        """
        count = mt5.positions_total()
        return count if count is not None else 0
    
    def get_positions(self, symbol: str = None) -> list:
        """
        Get list of open positions from MT5.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of position dictionaries
        """
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "magic": pos.magic,
                "comment": pos.comment
            })
        
        return result
    
    def close_position(self, ticket: int, volume: float = None) -> dict:
        """
        Close an open position by ticket.
        
        Args:
            ticket: Position ticket ID
            volume: Volume to close (optional, closes full position if None)
            
        Returns:
            Dict with 'ok' bool and error info if failed
        """
        # Get position info
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            return {"ok": False, "error": "position_not_found"}
        
        pos = positions[0]
        symbol = pos.symbol
        pos_volume = pos.volume if volume is None else volume
        pos_type = pos.type  # 0 = BUY, 1 = SELL
        
        # Determine close direction (opposite of position)
        if pos_type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        # Get correct filling mode for this symbol
        filling_mode = self.get_correct_filling_mode(symbol)
        
        # Build close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos_volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "REBEL close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        
        allow_fallback = True
        if symbol.upper() in NO_FILLING_MODE:
            request.pop("type_filling", None)
            allow_fallback = False
            print(f"[MT5] {symbol}: forcing broker default filling mode for close")

        result, _ = self._send_order_with_fill_fallback(request, allow_fallback=allow_fallback)
        
        if result is None:
            return {"ok": False, "error": "order_send_failed", "retcode": None}
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"ok": True, "order": result.order, "retcode": result.retcode}
        else:
            return {"ok": False, "error": result.comment, "retcode": result.retcode}
    
    def modify_position_sl(self, ticket: int, new_sl: float) -> dict:
        """
        Modify the stop loss of an open position.
        
        Args:
            ticket: Position ticket ID
            new_sl: New stop loss price
            
        Returns:
            Dict with 'ok' bool and error info if failed
        """
        # Get position info
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            return {"ok": False, "error": "position_not_found"}
        
        pos = positions[0]
        symbol = pos.symbol
        
        # Get symbol info for digits
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return {"ok": False, "error": "symbol_info_failed"}
        
        # Round SL to proper digits
        new_sl = round(new_sl, sym_info.digits)
        
        # Build modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": pos.tp,  # Keep existing TP
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            return {"ok": False, "error": "order_send_failed", "retcode": None}
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"ok": True, "retcode": result.retcode}
        else:
            return {"ok": False, "error": result.comment, "retcode": result.retcode}
    
    def can_trade(self) -> bool:
        """
        Check if trading is allowed in MT5.
        
        Verifies both terminal and account trading permissions.
        
        Returns:
            True if trading is allowed
        """
        info = mt5.terminal_info()
        acct = mt5.account_info()
        
        if info is None or acct is None:
            print("[BROKER:MT5] Unable to read terminal/account info. Blocking trading.")
            return False
        
        if not info.trade_allowed:
            print("[BROKER:MT5] Trading disabled in MT5 terminal (AutoTrading OFF).")
            return False
        
        if not acct.trade_allowed:
            print("[BROKER:MT5] Trading disabled for this account.")
            return False
        
        return True


# Allow running standalone for testing
if __name__ == "__main__":
    print("MT5Connector - Test Mode")
    
    connector = MT5Connector()
    
    if connector.initialize():
        print("\n--- Account Info ---")
        account = connector.get_account_info()
        if account:
            for key, val in account.items():
                print(f"  {key}: {val}")
        
        print("\n--- EURUSD Price ---")
        price = connector.get_current_price("EURUSD")
        if price:
            print(f"  Bid: {price['bid']}, Ask: {price['ask']}")
        
        print("\n--- EURUSD Symbol Info ---")
        info = connector.get_symbol_info("EURUSD")
        if info:
            for key, val in info.items():
                print(f"  {key}: {val}")
        
        print("\n--- Historical Data (M15, 10 bars) ---")
        df = connector.get_historical_data("EURUSD", "M15", 10)
        if df is not None:
            print(df)
        
        connector.shutdown()
    else:
        print("Failed to initialize MT5")

