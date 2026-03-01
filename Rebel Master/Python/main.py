"""
REBEL Trading Bot - Main Entry Point
Launches the engine, scanner, executor, and notifier for automated trading.
Supports pluggable broker connectors and strategy modes.
"""

import sys
import os
from datetime import datetime, timezone
import time

# Ensure the Python directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rebel_engine import RebelEngine
from rebel_scanner import RebelScanner
from rebel_trade_executor import RebelTradeExecutor
from rebel_notifier import RebelNotifier


# ============================================================================
#   MARKET TIME GUARD - Weekend Shutdown Control (DISABLED FOR ML TRAINING)
# ============================================================================

def market_time_allowed():
    """
    Controls when the REBEL SYSTEM is allowed to run.
    DISABLED: Running 24/7 for ML data collection.
    """
    # Weekend block DISABLED - run 7 days for ML training
    return True


def create_broker(config: dict):
    """
    Create a broker connector based on configuration.
    
    Args:
        config: Master configuration dictionary
        
    Returns:
        BrokerConnector instance
    """
    broker_config = config.get("broker", {})
    broker_type = broker_config.get("type", "mt5").lower()
    
    if broker_type == "mt5":
        from connectors.mt5_connector import MT5Connector
        return MT5Connector(config)
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")


def get_strategy_settings(config: dict) -> tuple:
    """
    Get strategy mode and minimum score from configuration.
    
    Args:
        config: Master configuration dictionary
        
    Returns:
        Tuple of (strategy_mode: str, min_score: int)
    """
    strategy_cfg = config.get("strategy", {})
    strategy_mode = strategy_cfg.get("mode", "normal")
    min_score_map = strategy_cfg.get("min_score", {
        "conservative": 4,
        "normal": 3,
        "aggressive": 2
    })
    min_score = min_score_map.get(strategy_mode, 3)
    
    return strategy_mode, min_score


def main():
    """Main entry point for the REBEL trading bot."""
    
    # ---- MARKET TIME GUARD ----
    if not market_time_allowed():
        print("[SYSTEM] Exiting due to weekend/time lock.")
        return 0
    
    print("=" * 50)
    print("  REBEL Trading Bot v1.0")
    print("  Universal Trading Engine")
    print("=" * 50)
    print()
    
    # Create engine instance
    engine = RebelEngine()
    
    # Load configuration
    try:
        config = engine.load_config()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[ERROR] Please ensure master_config.yaml exists in C:\\Rebel Technologies\\Rebel Master\\Config\\")
        return 1
    
    # Create broker connector
    try:
        broker = create_broker(config)
        engine.set_broker(broker)
        
        broker_type = config.get("broker", {}).get("type", "mt5")
        print(f"[MAIN] Broker: {broker_type.upper()}")
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1
    
    # Get strategy settings
    strategy_mode, min_score = get_strategy_settings(config)
    
    # Create components with shared broker and config
    scanner = RebelScanner(broker, config)
    executor = RebelTradeExecutor(broker, config)
    notifier = RebelNotifier(config)
    
    # Get scan interval from config (default: 60 seconds)
    interval = config.get("scanner", {}).get("interval", 60)
    
    # Display configuration summary
    print(f"[MAIN] Configuration Summary:")
    print(f"  - Strategy mode: {strategy_mode.upper()}")
    min_score_100 = int(round((min_score / 5.0) * 100))
    print(f"  - Min score: {min_score}/5 ({min_score_100}/100)")
    print(f"  - Scan interval: {interval}s")
    risk_cfg = config.get("risk_engine", {})
    print(f"  - Auto-trading: {'ENABLED' if executor.auto_trade else 'DISABLED'}")
    print(f"  - Risk per trade: {risk_cfg.get('percent_risk_per_trade', 3.0)}% (Rule 1)")
    print(f"  - Max exposure: {risk_cfg.get('max_equity_exposure_pct', 30.0)}% (Rule 2)")
    print(f"  - 1 position/symbol (Rule 3)")
    print(f"  - SL: {executor.sl_atr_multiplier}x ATR | TP: {executor.tp_atr_multiplier}x ATR")
    print(f"  - Telegram alerts: {'ENABLED' if notifier.enabled else 'DISABLED'}")
    print()
    
    print("[MAIN] Starting REBEL...")
    print("[MAIN] Press Ctrl+C to stop")
    print()
    
    # Run the engine with all components
    try:
        engine.run(
            scanner=scanner,
            executor=executor,
            notifier=notifier,
            interval=interval,
            min_score=min_score,
            strategy_mode=strategy_mode
        )
    except Exception as e:
        print(f"[ERROR] Engine error: {e}")
        if notifier.enabled:
            notifier.notify_error(str(e))
        return 1
    
    print("[MAIN] REBEL shutdown complete")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
