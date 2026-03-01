from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol


@dataclass(frozen=True)
class AccountInfo:
    login: int | str
    server: str
    balance: float
    currency: str


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class SymbolInfo:
    visible: bool
    volume_min: float
    volume_max: float
    volume_step: float
    trade_tick_size: float
    trade_tick_value: float
    digits: int


@dataclass(frozen=True)
class TickInfo:
    bid: float
    ask: float
    last: float
    time: int


class BrokerAdapter(Protocol):
    def connect(self, config: dict) -> bool: ...
    def shutdown(self) -> None: ...
    def last_error(self) -> Any: ...
    def get_timeframe(self, tf: str) -> Any: ...

    def account_info(self) -> Optional[AccountInfo]: ...
    def symbols_get(self) -> Optional[Iterable[Symbol]]: ...
    def symbol_select(self, symbol: str, enable: bool) -> bool: ...
    def symbol_info(self, symbol: str) -> Optional[SymbolInfo]: ...
    def symbol_info_tick(self, symbol: str) -> Optional[TickInfo]: ...
    def copy_rates_from_pos(self, symbol: str, timeframe: Any, start_pos: int, count: int) -> Any: ...
