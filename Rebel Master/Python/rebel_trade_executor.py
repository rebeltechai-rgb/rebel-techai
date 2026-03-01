"""
REBEL Trade Executor - Auto-Trading Module
Handles trade execution with ATR-based SL/TP, risk-based lot sizing, and CSV logging.
Broker-agnostic via connector interface.
"""

from datetime import datetime
import csv
import os
from typing import Optional, Dict, Any, TYPE_CHECKING
import MetaTrader5 as mt5

from rebel_risk import calculate_volume_3pct
from ml_trade_features import MLTradeFeatureLogger

if TYPE_CHECKING:
    from connectors.base_connector import BrokerConnector


# =============================================================================
# SPREAD FILTER WITH ATR FLOOR (prevents false skips on restart)
# =============================================================================
def is_spread_acceptable(symbol: str, spread: float, atr: float, spread_factor: float) -> bool:
    """
    Determines whether the current spread is acceptable based on ATR and floors.
    Prevents excessive skipping caused by tiny ATR values during low volatility.
    """
    # ----- BASE ATR FLOOR (FX) -----
    MIN_ATR = 0.0005

    sym = symbol.upper()

    # ----- METALS -----
    if sym.startswith(("XAU", "XAG", "XPT", "XPD")):
        MIN_ATR = 0.50

    # ----- INDICES -----
    elif sym.endswith(("500", "30", "2000", "100", "35", "20", "40", "225", "50")) or sym.endswith((".fs",)):
        MIN_ATR = 1.0

    # ----- CRYPTO (Axi available) -----
    elif sym.endswith(("USD", "-USD", "-JPY")) and sym[:3] in (
        "BTC", "ETH", "LTC", "BCH", "SOL", "ADA", "XRP", "XLM", "DOT", "DOG",
        "AVA", "AAV", "UNI", "SUS", "COM", "CRV", "LRC", "MAN", "SAN",
        "BAT", "BNB", "KSM", "XTZ", "LNK"):
        MIN_ATR = 10.0

    # Apply ATR floor
    atr = max(atr, MIN_ATR)

    # If still zero for any reason, accept trade
    if atr <= 0:
        return True

    spread_ratio = spread / atr
    return spread_ratio <= spread_factor


# =============================================================================
# BLOCKED SYMBOLS - Never trade these (broker issues, low liquidity, etc.)
# =============================================================================
BLOCKED_SYMBOLS = {
    "MKR-USD",    # Invalid volume / filling issues
    "MKRUSD",     # Invalid volume / filling issues
    "CHINA50",    # Broker rejects filling mode (10030)
    "CHINA50.FS", # Broker rejects filling mode (10030)
    "SAND-USD",   # Trades take weeks to close - capital trap
    "SANDUSD",    # Trades take weeks to close - capital trap
    "SOYBEAN.FS", # User-requested skip (poor trade quality)
    "SOYBEAN.fs", # Case variance
    "IRC",        # User-requested blacklist
    "IRC-USD",    # User-requested blacklist (crypto format)
    "IRCUSD",     # User-requested blacklist (no dash)
}

# =============================================================================
# RISK PROFILES BY SYMBOL TYPE (max lot caps) - TRAINING MODE
# Reduced lot sizes for ML data collection (revisit in ~1 week)
# =============================================================================
RISK_PROFILES = {
    "fx_major": {
        "max_lot": 0.05,  # FX majors only (EURUSD, GBPUSD, etc.)
    },
    "fx": {
        "max_lot": 0.01,  # FX minors, exotics, crosses
    },
    "indices": {
        "max_lot": 0.10,  # Stock indices
    },
    "metals": {
        "max_lot": 0.01,  # Gold, Silver, Platinum, Copper
    },
    "energies": {
        "max_lot": 0.10,  # Oil, Natural Gas
    },
    "crypto": {
        "max_lot": 1.0,   # Alt coins (ETH, SOL, LNK, etc.) - some have high min lots
    },
    "bnb": {
        "max_lot": 0.10,  # BNB training cap
    },
    "btc": {
        "max_lot": 0.10,  # Bitcoin only - lower cap
    },
    "softs": {
        "max_lot": 0.25,  # Cocoa, Coffee, Soybean
    },
    "default": {
        "max_lot": 0.01,  # Safety fallback
    },
}


def _classify_symbol(symbol: str) -> str:
    """Classify symbol into asset type for risk profiling."""
    s = symbol.upper()
    
    # BTC (check first - separate from other crypto for lower lot cap)
    if "BTC" in s:
        return "btc"
    
    # Crypto (alt coins)
    # Crypto tokens available on Axi MT5
    crypto_tokens = (
        "ETH", "XRP", "LTC", "ADA", "DOG", "DOT", "XLM", "SOL", 
        "AVAX", "AAVE", "BNB", "SAND", "UNI", "XTZ", "BCH", 
        "COMP", "CRV", "KSM", "LNK", "LRC", "MANA", "SUSHI", "BAT"
    )
    if "BNB" in s:
        return "bnb"
    if any(p in s for p in crypto_tokens):
        return "crypto"
    
    # Indices
    if any(k in s for k in ("US500", "US30", "US2000", "USTECH", "NAS100", "DAX40", 
                            "SPA35", "UK100", "HK50", "CHINA50", "AUS200", "EU50", 
                            "FRA40", "JPN225", "NETH25", "SWI20", "VIX", "USDINDEX",
                            "GER40", "IT40", "SGFREE", "CAC40", "EUSTX50", "HSI",
                            "NK225", "DJ30", "FT100", "SPI200", "S&P", "CN50")):
        return "indices"
    
    # Energies (Oil, Gas - separate from metals)
    if any(k in s for k in ("UKOIL", "USOIL", "BRENT", "WTI", "NATGAS", "OIL", "GAS")):
        return "energies"
    
    # Metals (Gold, Silver, Platinum, Copper)
    if any(k in s for k in ("XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER", "COPPER")):
        return "metals"
    
    # Softs
    if any(k in s for k in ("COCOA", "COFFEE", "SOYBEAN", "SUGAR", "COTTON", "WHEAT", "CORN")):
        return "softs"
    
    # FX: 6-letter pairs like EURUSD, GBPJPY etc.
    base = s.replace(".A", "").replace(".SA", "").replace(".FS", "").replace(".M", "")
    if len(base) == 6 and base.isalpha():
        # Check for FX majors (higher lot allowed)
        fx_majors = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD")
        if base in fx_majors:
            return "fx_major"
        return "fx"
    
    return "default"


def _apply_symbol_lot_cap(symbol: str, lot: float) -> float:
    """Apply symbol-specific lot cap based on risk profile."""
    profile = _classify_symbol(symbol)
    max_lot = RISK_PROFILES.get(profile, RISK_PROFILES["default"])["max_lot"]
    if lot > max_lot:
        print(f"[LOT] {symbol}: Capped from {lot:.2f} to {max_lot} ({profile} max)")
    return min(lot, max_lot)


def _apply_hard_lot_cap(symbol: str, lot: float, hard_cap: Optional[float]) -> float:
    """Apply a global hard lot cap if configured."""
    if hard_cap is None:
        return lot
    if lot > hard_cap:
        print(f"[LOT] {symbol}: Hard cap from {lot:.2f} to {hard_cap}")
    return min(lot, hard_cap)


class RebelTradeExecutor:
    """Trade execution class with risk management and logging."""
    
    def __init__(self, broker: "BrokerConnector", config: dict):
        """
        Initialize the trade executor with broker and configuration.
        
        Args:
            broker: BrokerConnector instance for trade execution
            config: Master configuration dictionary
        """
        self.broker = broker
        self.config = config
        self.trading_cfg = config.get("trading", {})
        ml_feature_cfg = config.get("ml_feature_logging", {})
        
        # Trading settings
        self.auto_trade = self.trading_cfg.get("auto_trade", False)
        self.kill_switch = bool(self.trading_cfg.get("kill_switch", False))
        self.kill_switch_drawdown_percent = float(
            self.trading_cfg.get("kill_switch_drawdown_percent", 0.0) or 0.0
        )
        self.force_min_lot = bool(self.trading_cfg.get("force_min_lot", False))
        self.sl_atr_multiplier = self.trading_cfg.get("sl_atr_multiplier", 2.0)
        self.tp_atr_multiplier = self.trading_cfg.get("tp_atr_multiplier", 3.0)
        self.max_spread_atr_ratio = self.trading_cfg.get("max_spread_atr_ratio", 0.30)
        
        # Training mode: global cap (set high since symbol-specific caps are now lower)
        # Symbol-specific caps in RISK_PROFILES take precedence
        self.training_lot_cap = self.trading_cfg.get("training_lot_cap", 1.0)
        self.hard_lot_cap = self.trading_cfg.get("hard_lot_cap")
        self.ml_feature_logging_enabled = bool(ml_feature_cfg.get("enabled", True))
        
        # Logging
        self.log_path = r"C:\Rebel Technologies\Rebel Master\logs\trades.csv"
        self.ml_feature_issue_log = r"C:\Rebel Technologies\Rebel Master\logs\ml_feature_issues.log"
        self._ensure_log_folder()
        
        # ML Trade Feature Logger (logs features at trade entry with ticket)
        self.ml_trade_logger = None
        if self.ml_feature_logging_enabled:
            self.ml_trade_logger = MLTradeFeatureLogger(base_path=r"C:\Rebel Technologies\Rebel Master\ML")
    
    def _ensure_log_folder(self):
        """Ensure the Logs folder exists."""
        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"[EXECUTOR] Created log directory: {log_dir}")

    def _is_drawdown_kill_switch_triggered(self) -> tuple[bool, str]:
        """Return (triggered, reason) based on configured drawdown kill switch."""
        if self.kill_switch_drawdown_percent <= 0:
            return False, ""
        account = mt5.account_info()
        if account is None:
            return False, "account_info_unavailable"
        balance = float(getattr(account, "balance", 0.0) or 0.0)
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        if balance <= 0:
            return False, "balance_unavailable"
        drawdown_pct = ((balance - equity) / balance) * 100.0
        if drawdown_pct >= self.kill_switch_drawdown_percent:
            reason = f"drawdown {drawdown_pct:.2f}% >= {self.kill_switch_drawdown_percent:.2f}%"
            return True, reason
        return False, ""

    def _log_ml_feature_issue(self, message: str) -> None:
        """Persist ML feature logging issues for later diagnosis."""
        try:
            with open(self.ml_feature_issue_log, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} | {message}\n")
        except Exception as e:
            print(f"[ML_LOG] Failed to write feature issue log: {e}")
    
    def _calc_lot(self, symbol: str, entry_price: float, sl_price: float) -> float:
        """
        Calculate lot size to risk exactly 3% of equity.
        Uses the standalone calculate_volume_3pct function from rebel_risk.
        """
        # Get account equity
        account = mt5.account_info()
        if account is None:
            raise RuntimeError("Failed to get account info from MT5")
        
        equity = account.equity
        
        # Get risk percentage from config
        risk_engine_cfg = self.config.get("risk_engine", {})
        risk_pct = risk_engine_cfg.get("percent_risk_per_trade", 3.0) / 100.0
        
        # Use the standalone function
        volume = calculate_volume_3pct(
            symbol=symbol,
            entry_price=entry_price,
            sl_price=sl_price,
            equity=equity,
            risk_pct=risk_pct,
        )
        
        if volume <= 0:
            print(f"[RISK] Cannot calculate safe 3% volume for {symbol}. Using min lot.")
            symbol_info = mt5.symbol_info(symbol)
            return symbol_info.volume_min if symbol_info else 0.01
        
        return volume
    
    def _log_trade(self, data: dict):
        """
        Log trade to CSV file.
        
        Args:
            data: Trade data dictionary
        """
        base_columns = [
            "timestamp", "symbol", "direction", "volume", "price",
            "sl", "tp", "score", "atr", "rsi", "adx",
            "ticket", "result", "comment"
        ]
        extra_columns = ["counter_trend"]
        columns = base_columns + extra_columns
        
        file_exists = os.path.exists(self.log_path)
        fieldnames = columns
        extrasaction = "raise"
        
        if file_exists:
            try:
                with open(self.log_path, "r", newline="") as rf:
                    reader = csv.reader(rf)
                    existing_header = next(reader, [])
                if existing_header:
                    if "counter_trend" not in existing_header:
                        fieldnames = existing_header
                        extrasaction = "ignore"
            except Exception:
                pass
        
        try:
            extra_fields = [k for k in data.keys() if k not in fieldnames]
            if extra_fields:
                extrasaction = "ignore"
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction=extrasaction)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(data)
        except Exception as e:
            print(f"[EXECUTOR] Failed to log trade: {e}")
    
    def open_trade(self, signal: dict) -> Dict[str, Any]:
        """
        Open a trade based on the signal.
        
        Args:
            signal: Signal dictionary from RebelSignals
            
        Returns:
            Dict with:
                - 'ok': bool indicating success
                - 'ticket': Order ticket or None
                - 'symbol': Trading symbol
                - 'direction': Trade direction
                - 'volume': Lot size
                - 'error': Error message if failed
        """
        # Extract signal data
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        indicators = signal.get("indicators", {})
        score = signal.get("score", 0)
        
        # Default result structure
        result = {
            "ok": False,
            "ticket": None,
            "symbol": symbol,
            "direction": direction,
            "volume": 0.0,
            "error": None
        }
        
        # Check blocked symbols list
        if symbol.upper() in BLOCKED_SYMBOLS or symbol in BLOCKED_SYMBOLS:
            print(f"[TRADE] {symbol} is BLOCKED — skipping")
            result["error"] = "symbol_blocked"
            return result
        
        # Check if auto-trading is enabled
        if not self.auto_trade:
            print("[TRADE] Auto-trade disabled, skipping live order")
            result["error"] = "auto_trade_disabled"
            return result

        # Drawdown-based kill switch (auto halt)
        drawdown_triggered, drawdown_reason = self._is_drawdown_kill_switch_triggered()
        if drawdown_triggered:
            print(f"[TRADE] Kill switch (drawdown) enabled, skipping live order: {drawdown_reason}")
            result["error"] = "kill_switch_drawdown"
            return result
        
        # Kill switch for emergency halt
        if self.kill_switch:
            print("[TRADE] Kill switch enabled, skipping live order")
            result["error"] = "kill_switch_enabled"
            return result
        
        if not symbol or not direction:
            print("[TRADE] Invalid signal: missing symbol or direction")
            result["error"] = "invalid_signal"
            return result
        
        # Ensure symbol is available
        if not self.broker.ensure_symbol(symbol):
            print(f"[TRADE] Symbol not available: {symbol}")
            result["error"] = "symbol_not_available"
            return result
        
        # Get symbol info
        info = self.broker.get_symbol_info(symbol)
        if info is None:
            print(f"[TRADE] Failed to get symbol info for {symbol}")
            result["error"] = "symbol_info_failed"
            return result
        
        # Get current price
        price_data = self.broker.get_current_price(symbol)
        if price_data is None:
            print(f"[TRADE] Failed to get price for {symbol}")
            result["error"] = "price_failed"
            return result
        
        # Determine entry price
        if direction == "long":
            price = price_data["ask"]
        else:
            price = price_data["bid"]
        
        # Get ATR for calculations
        atr = indicators.get("atr")
        rsi = indicators.get("rsi")
        adx = indicators.get("adx")
        
        # Check spread vs ATR ratio (with ATR floor to prevent false skips)
        if atr is not None:
            spread = self.broker.get_spread(symbol)
            if spread is not None:
                if not is_spread_acceptable(symbol, spread, atr, self.max_spread_atr_ratio):
                    result["error"] = "spread_too_high"
                    return result
        
        # Get point value for fallback calculations
        point = info.get("point", 0.0001)
        digits = info.get("digits", 5)
        
        # Get minimum stop level from broker (in points)
        # Use max of stops_level and freeze_level, some brokers use one or the other
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            stops_level = symbol_info.trade_stops_level or 0
            freeze_level = symbol_info.trade_freeze_level or 0
            min_stop_points = max(stops_level, freeze_level)
            spread_points = symbol_info.spread or 0
        else:
            min_stop_points = 0
            spread_points = 0
        
        # Convert to price distance and add spread + buffer
        min_stop_distance = (min_stop_points + spread_points) * point
        # Add 50% safety buffer for volatile exotics
        min_stop_distance *= 1.5
        
        # Absolute minimum: at least 20 points for any symbol
        absolute_min = 20 * point
        if min_stop_distance < absolute_min:
            min_stop_distance = absolute_min
        
        print(f"[STOPS] {symbol}: stops_level={min_stop_points}, spread={spread_points}, min_distance={min_stop_distance:.5f}")
        
        # Compute SL/TP distances
        if atr is not None and atr > 0:
            sl_distance = atr * self.sl_atr_multiplier
            tp_distance = atr * self.tp_atr_multiplier
        else:
            # Fallback to point-based defaults
            sl_distance = point * 50
            tp_distance = point * 100
        
        # Enforce minimum stop distance (10016 fix for exotics like USDIDR)
        if sl_distance < min_stop_distance:
            print(f"[SL] {symbol}: Expanding SL from {sl_distance:.5f} to min {min_stop_distance:.5f}")
            sl_distance = min_stop_distance
        if tp_distance < min_stop_distance:
            print(f"[TP] {symbol}: Expanding TP from {tp_distance:.5f} to min {min_stop_distance:.5f}")
            tp_distance = min_stop_distance
        
        # Calculate SL/TP prices
        if direction == "long":
            sl = price - sl_distance
            tp = price + tp_distance
        else:
            sl = price + sl_distance
            tp = price - tp_distance
        
        # Round to symbol digits
        price = round(price, digits)
        sl = round(sl, digits)
        tp = round(tp, digits)
        
        # Get account balance
        account = self.broker.get_account_info()
        balance = account.get("balance", 0) if account else 0
        
        if balance <= 0:
            print("[TRADE] Invalid account balance")
            result["error"] = "invalid_balance"
            return result
        
        # Calculate lot size (3% risk-based)
        try:
            volume = self._calc_lot(symbol, price, sl)
        except Exception as e:
            print(f"[TRADE] Lot calculation failed for {symbol}: {e}")
            result["error"] = f"lot_calc_error: {e}"
            return result
        
        if volume <= 0:
            print(f"[TRADE] Invalid lot size calculated for {symbol}")
            result["error"] = "invalid_lot_size"
            return result

        # Get symbol info for lot constraints (needed before scaling)
        symbol_info = mt5.symbol_info(symbol)
        vol_min = symbol_info.volume_min if symbol_info else 0.01
        vol_max = symbol_info.volume_max if symbol_info else 100.0
        vol_step = symbol_info.volume_step if symbol_info else 0.01

        # Soft-allow counter-trend trades at reduced size
        position_scale = signal.get("position_scale", 1.0)
        if isinstance(position_scale, (int, float)) and position_scale > 0 and position_scale < 1.0:
            scaled = volume * float(position_scale)
            print(f"[RISK] {symbol}: counter-trend scale {position_scale:.2f} applied ({volume:.4f} → {scaled:.4f})")
            volume = scaled

        # Force broker minimum lot size (micro units or broker min)
        if self.force_min_lot:
            print(f"[LOT] {symbol}: forcing broker min lot {vol_min:.4f}")
            volume = vol_min
        
        # Apply symbol-specific lot cap (indices, crypto, metals have lower caps)
        volume = _apply_symbol_lot_cap(symbol, volume)
        
        # Apply training lot cap (0.10 default for ML data collection)
        if volume > self.training_lot_cap:
            print(f"[TRAIN] {symbol}: Capped from {volume:.4f} to {self.training_lot_cap:.4f} (training mode)")
            volume = self.training_lot_cap
        
        # Apply hard global lot cap (absolute ceiling)
        volume = _apply_hard_lot_cap(symbol, volume, self.hard_lot_cap)
        
        # CHECK: Ensure volume is not below broker minimum (10014 fix)
        if volume < vol_min:
            if self.hard_lot_cap is not None and self.hard_lot_cap < vol_min:
                print(f"[LOT] {symbol}: Capped volume {volume:.4f} is below broker min {vol_min:.4f}")
                print(f"[LOT] {symbol}: BLOCKED — hard cap {self.hard_lot_cap:.4f} below minimum lot size")
                result["error"] = f"volume_below_min_{volume:.4f}_min_{vol_min:.4f}"
                return result
            # If counter-trend scaling pushed below min, use broker minimum instead of blocking.
            print(f"[LOT] {symbol}: Scaled volume {volume:.4f} below broker min {vol_min:.4f}, using min lot")
            volume = vol_min
        
        # Round to broker step
        volume = round(volume / vol_step) * vol_step
        volume = max(vol_min, min(vol_max, volume))
        
        result["volume"] = volume
        
        # --- MARGIN CHECK WITH AUTO-REDUCTION ---
        risk_engine_cfg = self.config.get("risk_engine", {})
        max_equity_exposure_pct = risk_engine_cfg.get("max_equity_exposure_pct", 30.0) / 100.0
        
        # Estimate margin for new trade
        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
        margin_required = mt5.order_calc_margin(order_type, symbol, volume, price)
        
        if margin_required is not None and margin_required > 0:
            equity = account.get("equity", 0)
            free_margin = account.get("free_margin", 0)
            
            # Current system exposure (correct Trading 101 formula)
            margin_used = equity - free_margin
            projected_margin = margin_used + margin_required
            projected_exposure = projected_margin / equity if equity > 0 else 1.0
            
            print(f"[MARGIN] {symbol}: Current=${margin_used:.2f}, New=${margin_required:.2f}, "
                  f"Projected={projected_exposure*100:.1f}% (limit={max_equity_exposure_pct*100:.0f}%)")
            
            if projected_exposure > max_equity_exposure_pct:
                # Auto-reduce: How much margin can we add and still be at max exposure?
                allowed_margin_total = max_equity_exposure_pct * equity
                remaining_margin_headroom = allowed_margin_total - margin_used
                
                if remaining_margin_headroom <= 0:
                    print(f"[RISK] BLOCKED {symbol} — no margin headroom left")
                    result["error"] = f"margin_limit_exceeded: {projected_exposure*100:.1f}%"
                    return result
                
                # Scale lot down to fit into remaining margin
                scale = remaining_margin_headroom / margin_required
                original_lot = volume
                scaled_lot = volume * max(0.0, min(1.0, scale))
                
                # Round to broker step
                import math
                scaled_lot = math.floor(scaled_lot / vol_step) * vol_step
                
                # Re-apply symbol cap for safety
                scaled_lot = _apply_symbol_lot_cap(symbol, scaled_lot)
                
                # Re-apply training cap for safety
                if scaled_lot > self.training_lot_cap:
                    scaled_lot = self.training_lot_cap

                # Re-apply hard cap for safety
                scaled_lot = _apply_hard_lot_cap(symbol, scaled_lot, self.hard_lot_cap)
                
                if scaled_lot < vol_min:
                    print(f"[RISK] BLOCKED {symbol} — scaled lot {scaled_lot:.2f} below minimum {vol_min}")
                    result["error"] = f"margin_limit_exceeded_after_scale: {projected_exposure*100:.1f}%"
                    return result
                
                print(f"[MARGIN] {symbol}: Requested lot={original_lot:.2f} → scaled to {scaled_lot:.2f} to respect {max_equity_exposure_pct*100:.0f}% exposure")
                volume = scaled_lot
                result["volume"] = volume
        
        # Place the order via broker
        comment = f"REBEL v1 score={score}"
        if signal.get("counter_trend"):
            comment += " counter_trend"
        broker_result = self.broker.place_market_order(
            symbol=symbol,
            direction=direction,
            volume=volume,
            sl=sl,
            tp=tp,
            comment=comment,
            extra={"magic": 20251202, "deviation": 10}
        )
        
        # Process result
        ticket = None
        result_code = broker_result.get("retcode")
        result_message = broker_result.get("message", "")
        
        if broker_result.get("ok"):
            ticket = broker_result.get("order")
            result["ok"] = True
            result["ticket"] = ticket
            print(f"[TRADE] {symbol} {direction.upper()} opened, ticket {ticket}, volume {volume}")
            
            # Log ML features at trade entry time (for ML training dataset)
            if self.ml_feature_logging_enabled and self.ml_trade_logger:
                features = signal.get("features", {})
                if features:
                    try:
                        self.ml_trade_logger.log_trade_features(
                            ticket=ticket,
                            symbol=symbol,
                            direction=direction,
                            entry_price=price,
                            sl=sl,
                            tp=tp,
                            volume=volume,
                            score=score,
                            features=features
                        )
                    except Exception as e:
                        msg = f"ticket={ticket} symbol={symbol} error={e}"
                        print(f"[ML_LOG] Failed to log trade features: {msg}")
                        self._log_ml_feature_issue(msg)
                else:
                    msg = f"ticket={ticket} symbol={symbol} missing_features"
                    print(f"[ML_LOG] Missing features for trade entry: {symbol} (ticket {ticket})")
                    self._log_ml_feature_issue(msg)
        else:
            result["error"] = f"broker_rejected: {result_code} - {result_message}"
            print(f"[TRADE] Order rejected for {symbol}: {result_code} - {result_message}")
        
        # Log the trade
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "direction": direction,
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp,
            "score": score,
            "atr": atr,
            "rsi": rsi,
            "adx": adx,
            "counter_trend": bool(signal.get("counter_trend")),
            "ticket": ticket,
            "result": result_code,
            "comment": result_message if not broker_result.get("ok") else "success"
        }
        self._log_trade(log_data)
        
        return result
