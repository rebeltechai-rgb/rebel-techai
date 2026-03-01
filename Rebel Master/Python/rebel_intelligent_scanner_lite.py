"""
REBEL Intelligent Scanner - Lite Intelligence Follower

This scanner:
- Connects to MT5 directly
- Reads OHLC data for symbols
- Uses RebelAI in "scanner" mode (reduced depth)
- NEVER places trades
- Prints signals and intelligence, designed for manual trading (e.g., AXI Select)

The main REBEL system should use RebelAI in "engine" mode, which is always
deeper and more powerful than this scanner mode.
"""

import time
from typing import List
import MetaTrader5 as mt5
import pandas as pd

from rebel_ai_core import RebelAI, RebelAIConfig


# ---------- Configuration ----------

SYMBOLS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "NZDUSD",
    "XAUUSD", "XAGUSD",
    "BTCUSD", "ETHUSD",
]

TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}

BARS = 400
SCAN_INTERVAL_SECONDS = 60


# ---------- Helper functions ----------

def _fetch_ohlc(symbol: str, timeframe, bars: int) -> pd.DataFrame | None:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        print(f"[SCANNER] Failed to fetch data for {symbol} on timeframe {timeframe}")
        return None
    df = pd.DataFrame(rates)
    df.rename(
        columns={
            "time": "time",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
        },
        inplace=True,
    )
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close"]]


def _ensure_symbol(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"[SCANNER] Symbol not found in MT5: {symbol}")
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True


# ---------- Scanner class ----------

class RebelIntelligentScanner:
    def __init__(self):
        self.ai = RebelAI(RebelAIConfig())

    def scan_symbol(self, symbol: str) -> dict | None:
        if not _ensure_symbol(symbol):
            return None

        df_m5 = _fetch_ohlc(symbol, TIMEFRAMES["M5"], BARS)
        df_m15 = _fetch_ohlc(symbol, TIMEFRAMES["M15"], BARS)
        df_h1 = _fetch_ohlc(symbol, TIMEFRAMES["H1"], BARS)

        if df_m5 is None or df_m15 is None or df_h1 is None:
            return None

        # IMPORTANT: scanner uses "scanner" mode, which is purposely weaker
        intel = self.ai.analyze(symbol, df_m5, df_m15, df_h1, mode="scanner")
        return intel

    def run_once(self):
        print("\n[INTEL-SCANNER] Running intelligence scan...")
        signals: list[dict] = []

        for symbol in SYMBOLS:
            intel = self.scan_symbol(symbol)
            if intel is None:
                continue

            score = intel["intelligence_score"]
            threshold = intel["adaptive_threshold"]
            bias = intel["bias"]

            # simple display filter: only show symbols where score is near/above threshold
            if score >= threshold - 5:
                signals.append(intel)

            self._print_symbol_intel(intel)

        print("\n[INTEL-SCANNER] Summary of strong candidates:")
        if not signals:
            print("  No high-confidence candidates this cycle.")
        else:
            for s in signals:
                print(
                    f"  {s['symbol']}: bias={s['bias']}, "
                    f"score={s['intelligence_score']}, "
                    f"threshold={s['adaptive_threshold']}"
                )

    def _print_symbol_intel(self, intel: dict):
        symbol = intel["symbol"]
        regime = intel["regime"]
        vol = intel["volatility"]
        mom = intel["momentum"]
        bias = intel["bias"]
        score = intel["intelligence_score"]
        threshold = intel["adaptive_threshold"]

        print(
            f"\n[INTEL] {symbol} | mode={intel['mode']}\n"
            f"  regime     : {regime}\n"
            f"  volatility : {vol}\n"
            f"  momentum   : {mom}\n"
            f"  bias       : {bias}\n"
            f"  score      : {score} / threshold {threshold}"
        )

        # Print only a couple of key reasons to keep it readable
        reasons = intel.get("reasons", [])
        if reasons:
            print("  reasons:")
            for r in reasons[:3]:
                print(f"    - {r}")
            if len(reasons) > 3:
                print(f"    - (+{len(reasons) - 3} more...)")

    def run_forever(self):
        print("==================================================")
        print("  REBEL Intelligent Scanner v3 (Lite Follower)")
        print("  Signal-only mode. NO TRADING.")
        print("==================================================")
        print(f"Scanning symbols: {', '.join(SYMBOLS)}")
        print(f"Interval: {SCAN_INTERVAL_SECONDS}s")
        print("Press Ctrl+C to stop.\n")

        while True:
            try:
                self.run_once()
                time.sleep(SCAN_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                print("\n[INTEL-SCANNER] Stopped by user.")
                break
            except Exception as e:
                print(f"[INTEL-SCANNER] Error in scan loop: {e}")
                time.sleep(5)


# ---------- entrypoint ----------

if __name__ == "__main__":
    if not mt5.initialize():
        print("[INTEL-SCANNER] Failed to initialize MT5")
    else:
        try:
            scanner = RebelIntelligentScanner()
            scanner.run_forever()
        finally:
            mt5.shutdown()

