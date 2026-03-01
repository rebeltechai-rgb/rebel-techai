"""
REBEL Base Connector - Abstract Broker Interface
Defines the interface that all broker connectors must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd


class BrokerConnector(ABC):
    """
    Abstract base class for broker connectors.
    
    All broker implementations must inherit from this class and implement
    all abstract methods to ensure compatibility with the REBEL trading engine.
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize connection to the broker.
        
        Returns:
            True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the broker connection and cleanup resources.
        """
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe string (e.g., 'M5', 'M15', 'H1')
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with columns: time, open, high, low, close, tick_volume
            or None if data unavailable
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current bid/ask prices for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with 'bid' and 'ask' keys, or None if unavailable
        """
        pass
    
    @abstractmethod
    def get_spread(self, symbol: str) -> Optional[float]:
        """
        Get current spread for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Spread in price terms, or None if unavailable
        """
        pass
    
    @abstractmethod
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
        Place a market order.
        
        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            volume: Lot size
            sl: Stop loss price (optional)
            tp: Take profit price (optional)
            comment: Order comment
            extra: Additional broker-specific parameters
            
        Returns:
            Dict with at least:
                - 'ok': bool indicating success
                - 'retcode': Broker return code
                - 'order': Order ticket/ID (if successful)
                - 'message': Error/status message
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current account information.
        
        Returns:
            Dict with account info including:
                - 'login': Account ID
                - 'balance': Account balance
                - 'equity': Account equity
                - 'margin': Used margin
                - 'free_margin': Available margin
                - 'leverage': Account leverage
                - 'profit': Current P/L
            or None if unavailable
        """
        pass
    
    @abstractmethod
    def ensure_symbol(self, symbol: str) -> bool:
        """
        Ensure a symbol is available for trading.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if symbol is available and ready, False otherwise
        """
        pass
    
    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with symbol info including:
                - 'point': Point value
                - 'digits': Price digits
                - 'trade_tick_value': Tick value
                - 'trade_tick_size': Tick size
                - 'volume_min': Minimum volume
                - 'volume_max': Maximum volume
                - 'volume_step': Volume step
            or None if unavailable
        """
        pass
    
    @abstractmethod
    def get_positions_count(self) -> int:
        """
        Get the number of open positions.
        
        Returns:
            Number of open positions
        """
        pass
    
    @abstractmethod
    def get_positions(self, symbol: str = None) -> list:
        """
        Get list of open positions.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of position dictionaries with at least:
                - 'ticket': Position ID
                - 'symbol': Trading symbol
                - 'type': 'buy' or 'sell'
                - 'volume': Position size
                - 'profit': Current P/L
        """
        pass
    
    def close_position(self, ticket: int, volume: float = None) -> dict:
        """
        Close an open position by ticket.
        
        Args:
            ticket: Position ticket ID
            volume: Volume to close (optional, closes full position if None)
            
        Returns:
            Dict with 'ok' bool and error info if failed
        """
        return {"ok": False, "error": "not_implemented"}
    
    def modify_position_sl(self, ticket: int, new_sl: float) -> dict:
        """
        Modify the stop loss of an open position.
        
        Args:
            ticket: Position ticket ID
            new_sl: New stop loss price
            
        Returns:
            Dict with 'ok' bool and error info if failed
        """
        return {"ok": False, "error": "not_implemented"}
    
    def can_trade(self) -> bool:
        """
        Check if trading is allowed on this broker.
        
        This is a safety check that verifies broker-level trading permissions
        (e.g., AutoTrading enabled in MT5). Broker-specific connectors should
        override this method with their own implementation.
        
        Returns:
            True if trading is allowed, False otherwise
        """
        return True

