from __future__ import annotations

from typing import Any, Iterable, Optional

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - runtime dependency check
    mt5 = None

from .base import AccountInfo, BrokerAdapter, Symbol, SymbolInfo, TickInfo


class MT5Adapter(BrokerAdapter):
    def __init__(self):
        self._connected = False
        self._last_error: Any = None

    def connect(self, config: dict) -> bool:
        if mt5 is None:
            self._last_error = ("MetaTrader5 package not installed",)
            return False
        adapter_cfg = config.get("broker_adapter", {})
        mt5_cfg = config.get("mt5", {})

        attach_only = adapter_cfg.get("attach_only", True)
        path = adapter_cfg.get("mt5_path") or mt5_cfg.get("path") or r"C:\mt5_scanner_live\terminal64.exe"

        # Always reset any existing connection first.
        mt5.shutdown()

        if attach_only:
            if not mt5.initialize(path=path):
                return False
            info = mt5.account_info()
            if info is None:
                return False
            self._connected = True
            return True

        if not mt5.initialize(path=path if path else None):
            return False

        login = mt5_cfg.get("login")
        password = mt5_cfg.get("password")
        server = mt5_cfg.get("server")
        if login and password and server:
            if not mt5.login(login=login, password=password, server=server):
                return False

        self._connected = True
        return True

    def shutdown(self) -> None:
        if mt5 is None:
            return
        if self._connected:
            mt5.shutdown()
            self._connected = False

    def last_error(self) -> Any:
        if mt5 is None:
            return self._last_error or ("MetaTrader5 package not installed",)
        return mt5.last_error()

    def get_timeframe(self, tf: str) -> Any:
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
        }
        return tf_map.get(tf.upper(), mt5.TIMEFRAME_H1)

    def account_info(self) -> Optional[AccountInfo]:
        info = mt5.account_info()
        if info is None:
            return None
        return AccountInfo(
            login=info.login,
            server=str(info.server),
            balance=float(info.balance),
            currency=str(info.currency),
        )

    def symbols_get(self) -> Optional[Iterable[Symbol]]:
        symbols = mt5.symbols_get()
        if not symbols:
            return None
        return [Symbol(name=s.name) for s in symbols]

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return bool(mt5.symbol_select(symbol, enable))

    def symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return SymbolInfo(
            visible=bool(info.visible),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            trade_tick_size=float(info.trade_tick_size),
            trade_tick_value=float(info.trade_tick_value),
            digits=int(info.digits),
        )

    def symbol_info_tick(self, symbol: str) -> Optional[TickInfo]:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return TickInfo(
            bid=float(tick.bid),
            ask=float(tick.ask),
            last=float(tick.last),
            time=int(tick.time),
        )

    def copy_rates_from_pos(self, symbol: str, timeframe: Any, start_pos: int, count: int) -> Any:
        return mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
