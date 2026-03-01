"""
ml_trade_logger.py

Logs ML shadow mode decisions for post-analysis.

Log format:
2025-12-10 14:23:51 | EURUSD | ML=REJECT | WinProb=0.32 LossProb=0.68 | Reasons=[...] | Engine=EXECUTED

Use this to validate:
- Did ML reject trades that became big losses? ✓
- Did ML accept trades that became big wins? ✓
- Did ML mistakenly reject winners? ✗
- Did ML miss toxic trades? ✗
"""

import os
import datetime

ML_LOG_PATH = r"C:\Rebel Technologies\Rebel Master\logs\ml_shadow_log.txt"


def _ensure_log_dir():
    """Ensure the Logs directory exists."""
    log_dir = os.path.dirname(ML_LOG_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def log_ml_decision(symbol: str, ml_result: dict, engine_action: str):
    """
    Logs ML shadow mode evaluation.
    
    Args:
        symbol: Trading symbol (e.g. 'EURUSD')
        ml_result: Dict from MLTradeFilter.evaluate_candidate() or shadow_mode_decision()
                   Expected keys: 'decision', 'win_prob', 'loss_prob', 'reasons'
        engine_action: What the engine actually did:
                       'EXECUTED' - Trade was opened
                       'SKIPPED'  - Trade skipped (other reason)
                       'BLOCKED'  - Trade blocked by ML filter
    """
    _ensure_log_dir()
    
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    line = (
        f"{ts} | {symbol} | "
        f"ML={ml_result['decision']} | "
        f"WinProb={ml_result['win_prob']:.2f} | "
        f"LossProb={ml_result['loss_prob']:.2f} | "
        f"Reasons={ml_result.get('reasons', [])} | "
        f"Engine={engine_action}\n"
    )
    
    with open(ML_LOG_PATH, "a") as f:
        f.write(line)


def log_ml_decision_simple(
    symbol: str,
    decision: str,
    win_prob: float,
    loss_prob: float,
    reasons: list,
    engine_action: str
):
    """
    Simplified logging without needing the full ml_result dict.
    
    Useful when you don't have an ML model loaded yet but want to
    log placeholder decisions during training data collection.
    """
    ml_result = {
        "decision": decision,
        "win_prob": win_prob,
        "loss_prob": loss_prob,
        "reasons": reasons
    }
    log_ml_decision(symbol, ml_result, engine_action)

