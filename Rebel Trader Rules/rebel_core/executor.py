"""
REBEL RULES-BASED AI TRADING SYSTEM
Module: Trade Executor
Handles order execution on MetaTrader 5
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5

# Set up logging
log = logging.getLogger("REBEL.Executor")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamps safely."""
    if not ts:
        return None
    try:
        # Handle Zulu time suffix if present
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


class TradeExecutor:
    """
    Trade execution engine for MetaTrader 5.
    """
    
    MAGIC_NUMBER = 777000  # Unique identifier for REBEL trades
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.log_dir = self.config.get('log_dir', 'logs')
        self.log_file = os.path.join(self.log_dir, 'trades.jsonl')
        self.dry_run = self.config.get('dry_run', False)
        self.trade_stats_file = os.path.join(self.log_dir, 'trade_family_stats.json')
        self.trade_counter_file = os.path.join(self.log_dir, 'trade_counter.json')
        self.governance_state_file = os.path.join(self.log_dir, 'governance_state.json')
        self.daily_counter_file = os.path.join(self.log_dir, 'daily_trade_counter.json')
        self.deal_history_state_file = os.path.join(self.log_dir, 'deal_history_state.json')
        self.position_groups_file = os.path.join(self.log_dir, 'position_groups.json')
        self.trade_counter_cfg = self.config.get('trade_counter', {})
        self.trade_counter_enabled = self.trade_counter_cfg.get('enabled', True)
        self.governance_cfg = self.config.get('governance', {})
        self.governance_enabled = self.governance_cfg.get('enabled', True)
        
        # Dry run tracking
        self.virtual_positions: List[Dict[str, Any]] = []
        self.virtual_positions_file = os.path.join(self.log_dir, 'virtual_positions.json')
        self.dry_run_stats = {"wins": 0, "losses": 0, "total_profit": 0.0}
        self.family_stats = {}
        self.family_groups = []
        self.symbol_groups = {}
        self.deal_history_state = {}
        self.position_groups = {}
        self.trade_counter = {}
        self.governance_state = {}
        self.daily_counter = {}
        self._live_closed_cache = {"count": 0, "mtime": None, "size": None}
        self.rules_version_id = self.config.get('rules_version_id', 'UNKNOWN_PRE_VERSION')
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        self.family_stats = self._load_trade_stats()
        self.deal_history_state = self._load_deal_history_state()
        self.position_groups = self._load_position_groups()
        self.trade_counter = self._load_trade_counter()
        self.governance_state = self._load_governance_state()
        self.daily_counter = self._load_daily_counter()

    def get_live_closed_outcome_count(self) -> int:
        """Return count of live CLOSED trades with outcomes in trades.jsonl."""
        try:
            if not os.path.exists(self.log_file):
                return 0
            stat = os.stat(self.log_file)
            mtime = stat.st_mtime
            size = stat.st_size
            cache = self._live_closed_cache
            if cache["mtime"] == mtime and cache["size"] == size:
                return int(cache.get("count", 0) or 0)
            count = 0
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if record.get("status") != "CLOSED":
                        continue
                    if record.get("dry_run"):
                        continue
                    if not record.get("outcome"):
                        continue
                    count += 1
            self._live_closed_cache = {"count": count, "mtime": mtime, "size": size}
            return count
        except Exception:
            return int(self._live_closed_cache.get("count", 0) or 0)

    def set_family_groups(self, groups: List[str]) -> None:
        """Seed family stats with known groups so they always display."""
        self.family_groups = [g.lower() for g in groups if g]
        for group in self.family_groups:
            if group not in self.family_stats:
                self.family_stats[group] = {"wins": 0, "losses": 0, "total": 0, "win_rate": 0.0}
        self._save_trade_stats()

    def set_symbol_groups(self, symbol_groups: Dict[str, str]) -> None:
        """Provide symbol -> group mapping for fallback classification."""
        self.symbol_groups = {str(sym).upper(): str(group).lower() for sym, group in symbol_groups.items()}
        
        # Load existing virtual positions if any
        if self.dry_run:
            self._load_virtual_positions()
    
    def execute_trade(
        self,
        symbol: str,
        direction: str,
        sl: Optional[float],
        tp: Optional[float],
        risk_scale: float = 1.0,
        comment: str = "REBEL",
        tags: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a trade on MT5.
        
        Args:
            symbol: Trading symbol
            direction: "BUY" or "SELL"
            sl: Stop loss price
            tp: Take profit price
            risk_scale: Position size multiplier
            comment: Trade comment
        
        Returns:
            Result dictionary with success status
        """
        if self.dry_run:
            return self._simulate_trade(symbol, direction, sl, tp, risk_scale, comment, tags=tags)
        
        if not self._init_mt5():
            return {"success": False, "error": "MT5 initialization failed"}
        
        try:
            # Validate symbol
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"success": False, "error": f"Symbol {symbol} not found"}
            
            # Ensure symbol is visible
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {"success": False, "error": f"Failed to select {symbol}"}
            
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"success": False, "error": f"No tick data for {symbol}"}
            
            # Determine order type and price
            if direction.upper() == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            elif direction.upper() == "SELL":
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                return {"success": False, "error": f"Invalid direction: {direction}"}
            
            # Calculate lot size
            lot_size = self._calculate_lot_size(symbol, price, sl, risk_scale)
            
            # Build order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": self.MAGIC_NUMBER,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._get_filling_mode(symbol)
            }
            
            # Add SL/TP
            if sl is not None:
                request["sl"] = float(sl)
            if tp is not None:
                request["tp"] = float(tp)
            
            # Execute (with filling-mode fallback)
            result = self._send_order_with_fill_fallback(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "success": False,
                    "error": f"Order failed: {result.comment}",
                    "retcode": result.retcode
                }
            
            # Log trade
            merged_tags = tags or {}
            trade_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "direction": direction,
                "volume": lot_size,
                "price": price,
                "sl": sl,
                "tp": tp,
                "ticket": result.order,
                "risk_scale": risk_scale,
                "comment": comment,
                "group": self._extract_group(comment),
                "decision_source": merged_tags.get("decision_source", "RULES"),
                "rules_version_id": self.rules_version_id,
                "tags": merged_tags
            }
            self._log_trade(trade_record)
            self._update_governance_counter(trade_record)
            
            return {
                "success": True,
                "ticket": result.order,
                "price": price,
                "volume": lot_size,
                "sl": sl,
                "tp": tp
            }
            
        except Exception as e:
            return {"success": False, "error": f"Exception: {str(e)}"}
        
        finally:
            mt5.shutdown()
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions for this EA (includes virtual positions in dry run mode)."""
        result = []
        
        # In dry run mode, include virtual positions for duplicate symbol checking
        if self.dry_run:
            for vpos in self.virtual_positions:
                if vpos.get("status") == "OPEN":
                    result.append({
                        "ticket": vpos.get("ticket", ""),
                        "symbol": vpos.get("symbol", ""),
                        "type": vpos.get("direction", ""),
                        "volume": vpos.get("volume", 0.01),
                        "price_open": vpos.get("price", 0),
                        "price_current": vpos.get("price", 0),
                        "sl": vpos.get("sl"),
                        "tp": vpos.get("tp"),
                        "profit": 0.0,
                        "comment": "VIRTUAL",
                        "magic": self.MAGIC_NUMBER
                    })
        
        # Also get real MT5 positions
        if not self._init_mt5():
            return result
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return result
            
            for pos in positions:
                # Only our trades (by magic number)
                if pos.magic == self.MAGIC_NUMBER or self.config.get('show_all_positions', False):
                    result.append({
                        "ticket": pos.ticket,
                        "symbol": pos.symbol,
                        "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                        "volume": pos.volume,
                        "price_open": pos.price_open,
                        "price_current": pos.price_current,
                        "sl": pos.sl,
                        "tp": pos.tp,
                        "profit": pos.profit,
                        "comment": pos.comment,
                        "magic": pos.magic
                    })
            
            return result
            
        except Exception:
            return result
        finally:
            mt5.shutdown()
    
    def close_position(self, ticket: int) -> Dict[str, Any]:
        """Close a specific position by ticket."""
        if not self._init_mt5():
            return {"success": False, "error": "MT5 initialization failed"}
        
        try:
            # Get position
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {"success": False, "error": f"Position {ticket} not found"}
            
            pos = position[0]
            
            # Get tick
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                return {"success": False, "error": f"No tick for {pos.symbol}"}
            
            # Close direction
            if pos.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
            # Close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": self.MAGIC_NUMBER,
                "comment": "REBEL_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._get_filling_mode(pos.symbol)
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"success": False, "error": f"Close failed: {result.comment}"}

            # Record stats by family (best-effort for manual closes)
            self._record_trade_outcome(pos.symbol, pos.comment, pos.profit, position_id=pos.ticket)
            
            return {
                "success": True,
                "ticket": ticket,
                "close_price": price,
                "profit": pos.profit
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            mt5.shutdown()
    
    def close_all_positions(self) -> Dict[str, Any]:
        """Close all positions (emergency)."""
        positions = self.get_open_positions()
        
        closed = 0
        errors = []
        
        for pos in positions:
            result = self.close_position(pos['ticket'])
            if result['success']:
                closed += 1
            else:
                errors.append(f"Ticket {pos['ticket']}: {result.get('error')}")
        
        return {
            "success": len(errors) == 0,
            "closed": closed,
            "total": len(positions),
            "errors": errors
        }
    
    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Modify SL/TP for an open position."""
        if not self._init_mt5():
            return {"success": False, "error": "MT5 initialization failed"}
        
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {"success": False, "error": f"Position {ticket} not found"}
            
            pos = position[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": sl if sl is not None else pos.sl,
                "tp": tp if tp is not None else pos.tp
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"success": False, "error": f"Modify failed: {result.comment}"}
            
            return {"success": True, "ticket": ticket, "sl": sl, "tp": tp}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            mt5.shutdown()
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self._init_mt5():
            return {}
        
        try:
            info = mt5.account_info()
            if info is None:
                return {}
            
            return {
                "balance": info.balance,
                "equity": info.equity,
                "margin": info.margin,
                "free_margin": info.margin_free,
                "profit": info.profit,
                "leverage": info.leverage,
                "currency": info.currency
            }
        except Exception:
            return {}
        finally:
            mt5.shutdown()
    
    def _init_mt5(self, max_retries: int = 5, retry_delay: int = 3) -> bool:
        """
        Initialize MT5 connection with retry logic.
        Handles 'resetting trading platform' state.
        """
        import time
        
        mt5_config = self.config.get('mt5', {})
        path = mt5_config.get('path')
        login = mt5_config.get('login')
        password = mt5_config.get('password')
        server = mt5_config.get('server')
        
        for attempt in range(1, max_retries + 1):
            try:
                # Shutdown any existing connection first
                try:
                    mt5.shutdown()
                except:
                    pass
                
                time.sleep(1)  # Let MT5 settle
                
                # Initialize with specific terminal path
                if path:
                    if not mt5.initialize(path):
                        print(f"[EXECUTOR] MT5 init failed (attempt {attempt}/{max_retries}): {mt5.last_error()}")
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                        continue
                else:
                    if not mt5.initialize():
                        print(f"[EXECUTOR] MT5 init failed (attempt {attempt}/{max_retries}): {mt5.last_error()}")
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                        continue
                
                # Wait for terminal to fully load (handles "resetting platform")
                time.sleep(2)
                
                # Verify terminal info with retries
                terminal_ready = False
                for check in range(3):
                    terminal_info = mt5.terminal_info()
                    if terminal_info is not None:
                        terminal_ready = True
                        break
                    print(f"[EXECUTOR] Waiting for terminal info... ({check+1}/3)")
                    time.sleep(2)
                
                if not terminal_ready:
                    print(f"[EXECUTOR] Terminal not ready (attempt {attempt}/{max_retries})")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                    continue
                
                # Check if logged into the CORRECT account
                account_info = mt5.account_info()
                if account_info is not None and login:
                    if account_info.login == login:
                        print(f"[EXECUTOR] MT5 connected on attempt {attempt} - Account {login}")
                        return True  # Already on correct account
                
                # Login with credentials
                if login and password and server:
                    if not mt5.login(login=login, password=password, server=server):
                        print(f"[EXECUTOR] MT5 login failed: {mt5.last_error()}")
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                        continue
                    print(f"[EXECUTOR] Logged into account {login} on attempt {attempt}")
                
                return True
                
            except Exception as e:
                print(f"[EXECUTOR] MT5 connection error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
        
        print(f"[EXECUTOR] Failed to connect to MT5 after {max_retries} attempts")
        return False
    
    def _get_filling_mode(self, symbol: str) -> int:
        """Get appropriate filling mode for symbol."""
        # Prefer broker-provided default if available
        try:
            info = mt5.symbol_info(symbol)
            if info is not None and getattr(info, "filling_mode", None) is not None:
                return info.filling_mode
        except Exception:
            pass
        
        # Symbols that need IOC (Immediate or Cancel) instead of FOK
        IOC_SYMBOLS = {
            "FT100.FS", "DJ30.FS", "S&P.FS", "NK225.FS", "CAC40.FS",
            "EUSTX50.FS", "SPI200.FS", "VIX.FS", "NATGAS.FS", "COPPER.FS",
            "DAX40.FS", "NAS100.FS", "US500.FS", "US30.FS"
        }
        
        symbol_upper = symbol.upper()
        if symbol_upper in IOC_SYMBOLS:
            return mt5.ORDER_FILLING_IOC
        
        # Default: FOK (Fill or Kill)
        return mt5.ORDER_FILLING_FOK

    def _send_order_with_fill_fallback(self, request: Dict[str, Any]) -> Any:
        """Send order, retrying with alternate filling modes if broker rejects."""
        preferred = request.get("type_filling")
        candidates = [preferred, mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
        seen = set()
        ordered = []
        for mode in candidates:
            if mode is None or mode in seen:
                continue
            seen.add(mode)
            ordered.append(mode)
        
        invalid_fill_code = getattr(mt5, "TRADE_RETCODE_INVALID_FILL", None)
        
        last_result = None
        for mode in ordered:
            request["type_filling"] = mode
            result = mt5.order_send(request)
            last_result = result
            if result is None:
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return result
            is_invalid_fill = (
                (invalid_fill_code is not None and result.retcode == invalid_fill_code)
                or ("Unsupported filling mode" in str(result.comment))
            )
            if is_invalid_fill:
                continue
            return result
        
        return last_result
    
    def _calculate_lot_size(
        self,
        symbol: str,
        price: float,
        sl: Optional[float],
        risk_scale: float
    ) -> float:
        """Calculate position size based on risk."""
        try:
            symbol_info = mt5.symbol_info(symbol)
            account_info = mt5.account_info()
            
            if symbol_info is None or account_info is None:
                return symbol_info.volume_min if symbol_info else 0.01
            
            # Default risk: 1% of balance
            risk_percent = self.config.get('risk_per_trade', 1.0) / 100
            risk_amount = account_info.balance * risk_percent * risk_scale
            
            # Calculate based on SL distance
            if sl is not None and price > 0:
                sl_distance = abs(price - sl)
                if sl_distance > 0:
                    tick_value = symbol_info.trade_tick_value
                    tick_size = symbol_info.trade_tick_size
                    
                    if tick_size > 0 and tick_value > 0:
                        risk_per_lot = (sl_distance / tick_size) * tick_value
                        if risk_per_lot > 0:
                            lot_size = risk_amount / risk_per_lot
                        else:
                            lot_size = symbol_info.volume_min
                    else:
                        lot_size = symbol_info.volume_min
                else:
                    lot_size = symbol_info.volume_min
            else:
                lot_size = symbol_info.volume_min
            
            # Round to valid step
            step = symbol_info.volume_step
            if step > 0:
                lot_size = round(lot_size / step) * step
            
            # Clamp to limits
            lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))
            
            return lot_size
            
        except Exception:
            return 0.01
    
    def _simulate_trade(
        self,
        symbol: str,
        direction: str,
        sl: Optional[float],
        tp: Optional[float],
        risk_scale: float,
        comment: str,
        tags: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Simulate a trade for dry run mode."""
        # Get current price from MT5
        entry_price = self._get_current_price(symbol, direction)
        
        trade_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": direction,
            "volume": 0.01 * risk_scale,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "ticket": f"SIM-{datetime.now().timestamp()}",
            "risk_scale": risk_scale,
            "dry_run": True,
            "status": "OPEN",
            "comment": comment,
            "group": self._extract_group(comment),
            "tags": tags or {}
        }
        
        # Add to virtual positions for tracking
        self.virtual_positions.append(trade_record)
        self._save_virtual_positions()
        self._log_trade(trade_record)
        
        print(f"    [DRY RUN] Virtual position opened: {direction} {symbol} @ {entry_price:.5f}")
        if sl:
            print(f"       SL: {sl:.5f} | TP: {f'{tp:.5f}' if tp else 'None'}")
        
        return {
            "success": True,
            "ticket": trade_record["ticket"],
            "price": entry_price,
            "volume": trade_record["volume"],
            "sl": sl,
            "tp": tp,
            "dry_run": True
        }
    
    def _get_current_price(self, symbol: str, direction: str = "BUY") -> float:
        """Get current price for a symbol."""
        try:
            if not mt5.initialize():
                return 0.0
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return 0.0
            return tick.ask if direction.upper() == "BUY" else tick.bid
        except Exception:
            return 0.0
    
    def monitor_virtual_positions(self) -> List[Dict[str, Any]]:
        """
        Check all virtual positions for SL/TP hits.
        Returns list of closed positions with outcomes.
        """
        if not self.dry_run or not self.virtual_positions:
            return []
        
        closed = []
        still_open = []
        
        for pos in self.virtual_positions:
            if pos.get("status") != "OPEN":
                continue
            
            symbol = pos["symbol"]
            direction = pos["direction"]
            entry_price = pos["price"]
            sl = pos.get("sl")
            tp = pos.get("tp")
            volume = pos.get("volume", 0.01)
            
            # Get current price
            current_price = self._get_current_price(symbol, "SELL" if direction == "BUY" else "BUY")
            
            if current_price == 0:
                still_open.append(pos)
                continue
            
            # Check SL/TP hits
            sl_hit = False
            tp_hit = False
            
            if direction.upper() == "BUY":
                if sl and current_price <= sl:
                    sl_hit = True
                if tp and current_price >= tp:
                    tp_hit = True
            else:  # SELL
                if sl and current_price >= sl:
                    sl_hit = True
                if tp and current_price <= tp:
                    tp_hit = True
            
            if sl_hit or tp_hit:
                # Calculate profit
                if direction.upper() == "BUY":
                    exit_price = sl if sl_hit else tp
                    pip_diff = exit_price - entry_price
                else:
                    exit_price = sl if sl_hit else tp
                    pip_diff = entry_price - exit_price
                
                # Rough profit calculation (simplified)
                profit = self._calculate_virtual_profit(symbol, pip_diff, volume)
                
                outcome = "LOSS" if sl_hit else "WIN"
                pos["status"] = "CLOSED"
                pos["exit_price"] = exit_price
                pos["profit"] = profit
                pos["outcome"] = outcome
                pos["close_time"] = datetime.now(timezone.utc).isoformat()
                
                # Update stats
                if outcome == "WIN":
                    self.dry_run_stats["wins"] += 1
                else:
                    self.dry_run_stats["losses"] += 1
                self.dry_run_stats["total_profit"] += profit
                self._record_trade_outcome(symbol, pos.get("comment"), profit)
                
                # Log the outcome
                emoji = "[WIN]" if outcome == "WIN" else "[LOSS]"
                hit_type = "TP" if tp_hit else "SL"
                print(f"\n{emoji} [DRY RUN] SIMULATED {outcome} on {symbol}")
                print(f"    {hit_type} hit | Entry: {entry_price:.5f} → Exit: {exit_price:.5f}")
                print(f"    Profit: ${profit:+.2f} | W/L: {outcome[0]}")
                print(f"    [STATS] Running: {self.dry_run_stats['wins']}W / {self.dry_run_stats['losses']}L | Total: ${self.dry_run_stats['total_profit']:+.2f}")
                
                # Structured log for parsing
                log.info(f"[DRY RUN] Virtual {'TP' if tp_hit else 'SL'} hit on {symbol} | "
                         f"Profit: ${profit:.2f} | W/L: {'W' if profit > 0 else 'L'}")
                
                closed.append(pos)
                self._log_trade(pos)
            else:
                still_open.append(pos)
        
        self.virtual_positions = still_open + [p for p in self.virtual_positions if p.get("status") != "OPEN"]
        self._save_virtual_positions()
        
        return closed
    
    def _calculate_virtual_profit(self, symbol: str, pip_diff: float, volume: float) -> float:
        """Calculate rough profit for virtual position."""
        try:
            if not mt5.initialize():
                return pip_diff * volume * 10000  # Rough estimate
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return pip_diff * volume * 10000
            
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size
            
            if tick_size > 0 and tick_value > 0:
                ticks = pip_diff / tick_size
                profit = ticks * tick_value * volume
                return round(profit, 2)
            
            return pip_diff * volume * 10000
        except Exception:
            return pip_diff * volume * 10000
    
    def get_virtual_positions(self) -> List[Dict[str, Any]]:
        """Get open virtual positions."""
        return [p for p in self.virtual_positions if p.get("status") == "OPEN"]
    
    def get_dry_run_stats(self) -> Dict[str, Any]:
        """Get dry run performance stats."""
        total = self.dry_run_stats["wins"] + self.dry_run_stats["losses"]
        win_rate = (self.dry_run_stats["wins"] / total * 100) if total > 0 else 0
        
        return {
            "wins": self.dry_run_stats["wins"],
            "losses": self.dry_run_stats["losses"],
            "total_trades": total,
            "win_rate": round(win_rate, 1),
            "total_profit": round(self.dry_run_stats["total_profit"], 2),
            "open_positions": len(self.get_virtual_positions())
        }

    def get_family_stats(self) -> Dict[str, Any]:
        """Get win/loss stats by family."""
        return self.family_stats
    
    def _load_virtual_positions(self) -> None:
        """Load virtual positions from file."""
        try:
            if os.path.exists(self.virtual_positions_file):
                with open(self.virtual_positions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.virtual_positions = data.get("positions", [])
                    self.dry_run_stats = data.get("stats", {"wins": 0, "losses": 0, "total_profit": 0.0})
                    print(f"[LOAD] Loaded {len(self.get_virtual_positions())} virtual positions")
        except Exception:
            pass
    
    def _save_virtual_positions(self) -> None:
        """Save virtual positions to file."""
        try:
            with open(self.virtual_positions_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "positions": self.virtual_positions,
                    "stats": self.dry_run_stats
                }, f, indent=2)
        except Exception:
            pass

    def _load_trade_stats(self) -> Dict[str, Any]:
        """Load per-family stats from file."""
        try:
            if os.path.exists(self.trade_stats_file):
                with open(self.trade_stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _save_trade_stats(self) -> None:
        """Save per-family stats to file."""
        try:
            with open(self.trade_stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.family_stats, f, indent=2)
        except Exception:
            pass

    def _load_deal_history_state(self) -> Dict[str, Any]:
        """Load MT5 deal history state from file."""
        try:
            if os.path.exists(self.deal_history_state_file):
                with open(self.deal_history_state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _save_deal_history_state(self) -> None:
        """Save MT5 deal history state to file."""
        try:
            with open(self.deal_history_state_file, 'w', encoding='utf-8') as f:
                json.dump(self.deal_history_state, f, indent=2)
        except Exception:
            pass

    def _load_position_groups(self) -> Dict[str, str]:
        """Load position -> group mapping from file."""
        try:
            if os.path.exists(self.position_groups_file):
                with open(self.position_groups_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _load_trade_counter(self) -> Dict[str, Any]:
        """Load or initialize the persistent trade counter."""
        if not self.trade_counter_enabled:
            return {}

        default_start = self.trade_counter_cfg.get("start_time")
        counter = {
            "start_time": default_start or datetime.now(timezone.utc).isoformat(),
            "target_trades": self.trade_counter_cfg.get("target_trades", 1500),
            "total_trades": 0,
            "last_ticket": 0,
            "last_timestamp": None
        }

        if os.path.exists(self.trade_counter_file):
            try:
                with open(self.trade_counter_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        counter.update(data)
            except Exception:
                pass

        if self.trade_counter_cfg.get("rebuild_from_logs", True):
            counter = self._rebuild_trade_counter(counter)
            self._save_trade_counter(counter)

        return counter

    def _rebuild_trade_counter(self, counter: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild trade counter from trades.jsonl using start_time."""
        start_time = counter.get("start_time")
        try:
            start_dt = datetime.fromisoformat(start_time) if start_time else None
            if start_dt and start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
        except Exception:
            start_dt = None

        total = 0
        last_ticket = 0
        last_ts = counter.get("last_timestamp")

        if not os.path.exists(self.log_file):
            counter.update({"total_trades": total, "last_ticket": last_ticket, "last_timestamp": last_ts})
            return counter

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if record.get("dry_run"):
                        continue
                    ts = record.get("timestamp")
                    if ts and start_dt:
                        try:
                            rec_dt = datetime.fromisoformat(ts)
                            if rec_dt.tzinfo is None:
                                rec_dt = rec_dt.replace(tzinfo=timezone.utc)
                            if rec_dt < start_dt:
                                continue
                        except Exception:
                            pass
                    total += 1
                    ticket = record.get("ticket")
                    if isinstance(ticket, int):
                        last_ticket = max(last_ticket, ticket)
                    last_ts = ts or last_ts
        except Exception:
            pass

        counter.update({"total_trades": total, "last_ticket": last_ticket, "last_timestamp": last_ts})
        return counter

    def _save_trade_counter(self, counter: Optional[Dict[str, Any]] = None) -> None:
        """Persist trade counter to disk."""
        if not self.trade_counter_enabled:
            return
        data = counter if counter is not None else self.trade_counter
        try:
            with open(self.trade_counter_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_daily_counter(self) -> Dict[str, Any]:
        """Load or initialize daily trade counter (UTC)."""
        today = datetime.now(timezone.utc).date().isoformat()
        counter = {"date": today, "trades_today": 0}
        if os.path.exists(self.daily_counter_file):
            try:
                with open(self.daily_counter_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        counter.update(data)
            except Exception:
                pass
        if counter.get("date") != today:
            counter["date"] = today
            counter["trades_today"] = 0
            self._save_daily_counter(counter)
        return counter

    def _save_daily_counter(self, counter: Optional[Dict[str, Any]] = None) -> None:
        data = counter if counter is not None else self.daily_counter
        try:
            with open(self.daily_counter_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _update_daily_counter(self, trade_record: Dict[str, Any]) -> None:
        if trade_record.get("dry_run"):
            return
        ts = trade_record.get("timestamp")
        dt = _parse_iso(ts) if ts else None
        if dt is None:
            dt = datetime.now(timezone.utc)
        date_str = dt.date().isoformat()
        if self.daily_counter.get("date") != date_str:
            self.daily_counter["date"] = date_str
            self.daily_counter["trades_today"] = 0
        self.daily_counter["trades_today"] = int(self.daily_counter.get("trades_today", 0) or 0) + 1
        self._save_daily_counter()

    def get_daily_trade_counter(self) -> Dict[str, Any]:
        return dict(self.daily_counter)

    def _load_governance_state(self) -> Dict[str, Any]:
        """Load or initialize governance trade counter state."""
        if not self.governance_enabled:
            return {}

        lock_time = self.governance_cfg.get("lock_time")
        state = {
            "lock_time": lock_time,
            "target_trades": self.governance_cfg.get("target_trades", 1500),
            "governance_trade_count": 0,
            "per_family_counts": {},
            "last_trade_time": None,
            "phase": "PHASE_1"
        }

        if os.path.exists(self.governance_state_file):
            try:
                with open(self.governance_state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        state.update(data)
            except Exception:
                pass

        # Rebuild from logs to stay aligned with governed trades per family.
        if self.governance_cfg.get("rebuild_from_logs", True):
            state = self._rebuild_governance_state(state)

        # Normalize phase on load in case thresholds changed.
        state["phase"] = self._compute_governance_phase(state.get("governance_trade_count", 0))
        self._save_governance_state(state)
        return state

    def _rebuild_governance_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild governance counts from trade logs using lock_time."""
        lock_time = state.get("lock_time")
        if not lock_time or not os.path.exists(self.log_file):
            return state

        total = 0
        per_family = {}
        last_ts = None

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if record.get("dry_run"):
                        continue
                    trade_time = record.get("timestamp")
                    if not trade_time or not self._governance_lock_allows(trade_time, lock_time=lock_time):
                        continue

                    total += 1
                    last_ts = trade_time
                    group = (
                        record.get("group")
                        or self._extract_group(record.get("comment"))
                        or self._resolve_group_by_symbol(record.get("symbol"))
                        or "unknown"
                    )
                    group = str(group).lower()
                    per_family[group] = per_family.get(group, 0) + 1
        except Exception:
            return state

        state["governance_trade_count"] = total
        state["per_family_counts"] = per_family
        state["last_trade_time"] = last_ts
        return state

    def _save_governance_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        """Persist governance state to disk."""
        if not self.governance_enabled:
            return
        data = state if state is not None else self.governance_state
        try:
            with open(self.governance_state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _compute_governance_phase(self, count: int) -> str:
        if count < 500:
            return "PHASE_1"
        if count < 750:
            return "PHASE_2"
        if count < 1000:
            return "PHASE_3"
        if count < 1500:
            return "PHASE_4"
        return "DEPLOY_READY"

    def _governance_lock_allows(self, trade_time_iso: str, lock_time: Optional[str] = None) -> bool:
        lock_time = lock_time or self.governance_state.get("lock_time")
        if not lock_time:
            return True
        try:
            trade_time = datetime.fromisoformat(trade_time_iso)
            if trade_time.tzinfo is None:
                trade_time = trade_time.replace(tzinfo=timezone.utc)
            lock_dt = datetime.fromisoformat(lock_time)
            if lock_dt.tzinfo is None:
                lock_dt = lock_dt.replace(tzinfo=timezone.utc)
            return trade_time >= lock_dt
        except Exception:
            return True

    def _update_governance_counter(self, trade_record: Dict[str, Any]) -> None:
        """Increment governance counter for new opened trades after lock time."""
        if not self.governance_enabled:
            return
        if trade_record.get("dry_run"):
            return
        trade_time = trade_record.get("timestamp")
        if not trade_time or not self._governance_lock_allows(trade_time):
            return

        self.governance_state["governance_trade_count"] = int(
            self.governance_state.get("governance_trade_count", 0) or 0
        ) + 1
        family_counts = self.governance_state.setdefault("per_family_counts", {})
        group = (
            trade_record.get("group")
            or self._extract_group(trade_record.get("comment"))
            or self._resolve_group_by_symbol(trade_record.get("symbol"))
            or "unknown"
        )
        group = str(group).lower()
        family_counts[group] = int(family_counts.get(group, 0) or 0) + 1
        self.governance_state["last_trade_time"] = trade_time
        self.governance_state["phase"] = self._compute_governance_phase(
            self.governance_state["governance_trade_count"]
        )
        self._save_governance_state()
        print(
            f"[GOV] Incremented: {self.governance_state['governance_trade_count']} | "
            f"Phase: {self.governance_state['phase']}"
        )

    def get_governance_state(self) -> Dict[str, Any]:
        """Get current governance counter state."""
        return dict(self.governance_state) if self.governance_enabled else {}

    def _update_trade_counter(self, trade_record: Dict[str, Any]) -> None:
        """Increment trade counter for real trades."""
        if not self.trade_counter_enabled:
            return
        if trade_record.get("dry_run"):
            return
        ticket = trade_record.get("ticket")
        if isinstance(ticket, int) and ticket <= int(self.trade_counter.get("last_ticket", 0) or 0):
            return

        self.trade_counter["total_trades"] = int(self.trade_counter.get("total_trades", 0) or 0) + 1
        if isinstance(ticket, int):
            self.trade_counter["last_ticket"] = ticket
        self.trade_counter["last_timestamp"] = trade_record.get("timestamp") or self.trade_counter.get("last_timestamp")
        self._save_trade_counter()

    def get_trade_counter(self) -> Dict[str, Any]:
        """Get current trade counter state."""
        return dict(self.trade_counter) if self.trade_counter_enabled else {}

    def _save_position_groups(self) -> None:
        """Save position -> group mapping to file."""
        try:
            with open(self.position_groups_file, 'w', encoding='utf-8') as f:
                json.dump(self.position_groups, f, indent=2)
        except Exception:
            pass

    def _extract_group(self, comment: Optional[str]) -> str:
        """Extract group name from trade comment."""
        if not comment:
            return "unknown"
        token = comment.strip().upper()
        if token.startswith("[") and token.endswith("]"):
            token = token[1:-1].strip()
        if token.startswith("REBEL_"):
            token = token[len("REBEL_"):]
        token = token.replace(" ", "_")
        if not token:
            return "unknown"
        if token in ("CLOSE", "REBEL_CLOSE", "REBEL_CLOSE_ALL"):
            return "unknown"
        if token.startswith("SL") or token.startswith("TP") or token.startswith("SL_") or token.startswith("TP_"):
            return "unknown"
        return token.lower()

    def _resolve_group_by_symbol(self, symbol: Optional[str]) -> Optional[str]:
        """Fallback: infer group from symbol -> group map."""
        if not symbol:
            return None
        return self.symbol_groups.get(str(symbol).upper())

    def _record_trade_outcome(
        self,
        symbol: str,
        comment: Optional[str],
        profit: float,
        group_override: Optional[str] = None,
        position_id: Optional[int] = None
    ) -> None:
        """Record win/loss by family based on trade comment or override."""
        group = group_override.lower() if group_override else self._extract_group(comment)
        if group == "unknown" and position_id is not None:
            mapped = self.position_groups.get(str(position_id))
            if mapped:
                group = mapped
        if group == "unknown":
            mapped = self._resolve_group_by_symbol(symbol)
            if mapped:
                group = mapped
        if group not in self.family_stats:
            self.family_stats[group] = {"wins": 0, "losses": 0, "total": 0, "win_rate": 0.0}
        stats = self.family_stats[group]
        if profit > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["total"] = stats["wins"] + stats["losses"]
        stats["win_rate"] = round((stats["wins"] / stats["total"]) * 100, 1) if stats["total"] else 0.0
        self._save_trade_stats()

    def _resolve_group_for_deal(self, deal) -> str:
        """Resolve group for a deal using comment or position mapping."""
        group = self._extract_group(getattr(deal, "comment", None))
        if group != "unknown":
            return group
        position_id = getattr(deal, "position", None)
        if position_id is not None:
            mapped = self.position_groups.get(str(position_id))
            if mapped:
                return mapped
        mapped = self._resolve_group_by_symbol(getattr(deal, "symbol", None))
        if mapped:
            return mapped
        return "unknown"

    def sync_deal_history(self, lookback_minutes: int = 1440) -> Dict[str, Any]:
        """
        Sync MT5 deal history and update per-family stats for closed trades.
        Returns summary dict with new deal count.
        """
        if not self._init_mt5():
            return {"success": False, "error": "MT5 initialization failed", "new_deals": 0}

        now = datetime.now(timezone.utc)
        last_time_str = self.deal_history_state.get("last_time")
        last_ticket = int(self.deal_history_state.get("last_ticket", 0) or 0)

        if last_time_str:
            try:
                last_time = datetime.fromisoformat(last_time_str)
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
            except Exception:
                last_time = now - timedelta(minutes=lookback_minutes)
        else:
            last_time = now - timedelta(minutes=lookback_minutes)

        # Pull a little overlap to avoid missing edge cases
        from_time = last_time - timedelta(minutes=1)
        to_time = now

        try:
            deals = mt5.history_deals_get(from_time, to_time)
            if deals is None:
                return {"success": True, "new_deals": 0}

            new_count = 0
            newest_time = last_time
            newest_ticket = last_ticket

            for deal in sorted(deals, key=lambda d: (d.time, d.ticket)):
                if getattr(deal, "magic", None) != self.MAGIC_NUMBER:
                    continue

                deal_time = datetime.fromtimestamp(deal.time, timezone.utc)
                if deal_time < last_time:
                    continue
                if deal_time == last_time and deal.ticket <= last_ticket:
                    continue

                entry_type = getattr(deal, "entry", None)
                if entry_type == mt5.DEAL_ENTRY_IN:
                    group = self._extract_group(getattr(deal, "comment", None))
                    position_id = getattr(deal, "position", None)
                    if group != "unknown" and position_id is not None:
                        self.position_groups[str(position_id)] = group
                        self._save_position_groups()
                elif entry_type in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY):
                    group = self._resolve_group_for_deal(deal)
                    profit = float(getattr(deal, "profit", 0.0))
                    outcome = "WIN" if profit > 0 else "LOSS"
                    close_price = float(getattr(deal, "price", 0.0) or 0.0)
                    self._record_trade_outcome(
                        getattr(deal, "symbol", ""),
                        None,
                        profit,
                        group_override=group,
                        position_id=getattr(deal, "position", None)
                    )
                    # Persist live outcomes to trades.jsonl for RF readiness
                    self._log_closed_trade({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": getattr(deal, "symbol", ""),
                        "direction": "CLOSE",
                        "ticket": getattr(deal, "position", None) or getattr(deal, "ticket", None),
                        "group": group,
                        "dry_run": False,
                        "status": "CLOSED",
                        "exit_price": close_price,
                        "profit": profit,
                        "outcome": outcome,
                        "close_time": deal_time.isoformat(),
                        "decision_source": "RULES",
                        "rules_version_id": self.rules_version_id,
                    })
                    new_count += 1

                if deal_time > newest_time or (deal_time == newest_time and deal.ticket > newest_ticket):
                    newest_time = deal_time
                    newest_ticket = deal.ticket

            if new_count > 0:
                self.deal_history_state["last_time"] = newest_time.isoformat()
                self.deal_history_state["last_ticket"] = newest_ticket
                self._save_deal_history_state()

            return {"success": True, "new_deals": new_count}
        except Exception as e:
            return {"success": False, "error": str(e), "new_deals": 0}
        finally:
            mt5.shutdown()
    
    def get_deals_between(self, start_time: datetime, end_time: datetime) -> List[Any]:
        """Get MT5 deals between timestamps for this EA."""
        if not self._init_mt5():
            return []
        try:
            deals = mt5.history_deals_get(start_time, end_time)
            if not deals:
                return []
            return [d for d in deals if getattr(d, "magic", None) == self.MAGIC_NUMBER]
        except Exception:
            return []
        finally:
            mt5.shutdown()
    def _log_trade(self, trade_record: Dict[str, Any]) -> None:
        """Log trade to file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(trade_record, f)
                f.write('\n')
            self._update_trade_counter(trade_record)
            self._update_daily_counter(trade_record)
        except Exception:
            pass  # Silent fail - don't crash engine

    def _log_closed_trade(self, record: Dict[str, Any]) -> None:
        """Append a closed-trade record to trades.jsonl (live outcomes)."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(record, f)
                f.write('\n')
        except Exception:
            pass


def create_executor(config: Dict[str, Any] = None) -> TradeExecutor:
    """Factory function to create a TradeExecutor."""
    return TradeExecutor(config)


