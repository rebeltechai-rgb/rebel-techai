"""
REBEL RULES-BASED AI TRADING SYSTEM
Module: Main Engine
Orchestrates all components
"""

import csv
import os
import sys
import json
import time
import yaml
import traceback
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import pytz

from .indicators import compute_all_indicators, get_current_session
from .regimes.metals_impulse import (
    build_session_stats,
    detect_metals_impulse,
    get_session_window,
    previous_session_name,
)
from .scorer import SignalScorer
from .ai_brain import AIBrain
from .executor import TradeExecutor
from .risk_manager import RiskManager, ProfitLock
from .scanner_bridge import ScannerBridge

# Intelligent Scanner imports (Trader's own copy)
_SCANNER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Scanner")
if _SCANNER_DIR not in sys.path:
    sys.path.insert(0, _SCANNER_DIR)
try:
    from rebel_intelligent_scanner import RebelIntelligentScanner
    HAS_INTELLIGENT_SCANNER = True
except ImportError as e:
    print(f"[WARN] Intelligent Scanner not available: {e}")
    HAS_INTELLIGENT_SCANNER = False


class RebelEngine:
    """
    Main trading engine orchestrator.
    
    Coordinates:
    - Technical analysis (indicators)
    - Signal scoring (rules-based)
    - AI decision making (GPT with rules)
    - Trade execution (MT5)
    - Risk management
    """
    
    def __init__(self, config_path: str = "config/rebel_config.yaml"):
        self.config = self._load_config(config_path)
        self.running = False
        self.governance_state = {}
        self.governance_targets = self._compute_governance_targets(0)
        self.gpt_mode = "off"
        
        # Initialize components
        self.scorer = SignalScorer(self.config.get('scoring', {}))
        # Pass both execution and mt5 config to executor
        executor_config = self.config.get('execution', {})
        executor_config['mt5'] = self.config.get('mt5', {})
        executor_config['rules_version_id'] = self.config.get('rules_version_id', 'RT_RULES_v1_0_0')
        self.executor = TradeExecutor(executor_config)
        self.risk_manager = RiskManager(self.config.get('risk', {}))
        self.profit_lock = ProfitLock(self.config.get('profit_lock', {}))
        
        # AI Brain (optional - can run without)
        self.ai_brain = None
        if self.config.get('ai', {}).get('enabled', True):
            try:
                self.ai_brain = AIBrain(self.config.get('ai', {}))
                print("[OK] AI Brain initialized")
            except Exception as e:
                print(f"[!] AI Brain not available: {e}")
                print("  System will use score-based decisions only")
        
        # Scanner Bridge (optional - integrates Intelligent Scanner)
        self.scanner = None
        if self.config.get('scanner', {}).get('enabled', False):
            try:
                print("Loading Scanner Bridge...")
                self.scanner = ScannerBridge(self.config.get('scanner', {}))
                if self.scanner.scanner_available:
                    print("[OK] Scanner Bridge active")
                else:
                    print("[!] Scanner Bridge loaded but helpers unavailable")
                    self.scanner = None
            except Exception as e:
                print(f"[!] Scanner Bridge not available: {e}")
        
        # Intelligent Scanner (confidence gate layer — same pattern as Master)
        is_cfg = self.config.get("intelligent_scanner", {}) or {}
        self.intelligent_scanner_enabled = bool(is_cfg.get("enabled", False)) and HAS_INTELLIGENT_SCANNER
        self.intelligent_scanner = None
        self.scanner_session_confidence = is_cfg.get("session_confidence", {})
        self.scanner_global_min_conf = int(is_cfg.get("min_confidence", 65))
        if self.intelligent_scanner_enabled:
            try:
                self.intelligent_scanner = RebelIntelligentScanner(self.config)
                self.intelligent_scanner.adapter._connected = True
                self.intelligent_scanner.connected = True
                print(f"[SCANNER] Intelligent Scanner ACTIVE (min_conf={self.scanner_global_min_conf}%)")
            except Exception as e:
                print(f"[SCANNER] Failed to initialize Intelligent Scanner: {e}")
                self.intelligent_scanner_enabled = False

        # R:R Blocker state (per symbol+session cumulative R:R after N trades)
        rr_cfg = self.config.get("rr_blocker", {}) or {}
        self.rr_blocker_enabled = bool(rr_cfg.get("enabled", False))
        self.rr_blocker_min_trades = int(rr_cfg.get("min_trades", 30))
        self._rr_blocker_cache = {}
        self._rr_blocker_cache_ts = None

        # Build symbol list
        self.symbols = self._build_symbol_list()
        self.executor.set_family_groups(self.config.get('enabled_groups', []))
        self.regime_state = {"metals_impulse": False}
        self.session_stats = {}
        self.last_session = None
        self._last_rf_progress_date = None
        self._shadow_log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "gpt_shadow_log.jsonl"
        )
        self._rf_features_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "rf_features.csv"
        )
        self.rules_version_id = self.config.get("rules_version_id", "v1.0")
        self.executor.set_symbol_groups(self.symbols)
        
        print(f"\n{'='*60}")
        print("REBEL RULES-BASED AI TRADING SYSTEM")
        print(f"{'='*60}")
        print(f"Rules Version: {self.rules_version_id}")
        print(f"Symbols: {len(self.symbols)}")
        print(f"AI Mode: {'Enabled' if self.ai_brain else 'Disabled (score-only)'}")
        if self.ai_brain:
            ai_cfg = self.config.get("ai", {})
            deploy_at = int(ai_cfg.get("gpt_deploy_start", 1750))
            print(f"GPT Shadow: ACTIVE — logging to gpt_shadow_log.jsonl")
            print(f"GPT Deploy: At {deploy_at} governed trades GPT decides on its own")
        is_label = "Integrated" if self.scanner else ("Confidence Gate" if self.intelligent_scanner_enabled else "Standalone only")
        print(f"Scanner: {is_label}")
        if self.rr_blocker_enabled:
            print(f"R:R Blocker: ACTIVE (min_trades={self.rr_blocker_min_trades})")
        print(f"Dry Run: {self.config.get('execution', {}).get('dry_run', False)}")
        print(f"{'='*60}\n")
    
    def run(self) -> None:
        """Main engine loop."""
        self.running = True
        scan_interval = self.config.get('scan_interval', 60)
        
        print(f"Starting main loop (interval: {scan_interval}s)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                self._run_scan_cycle()
                
                print(f"\n[...] Waiting {scan_interval}s until next scan...")
                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            print("\n\n[STOP] Stopped by user")
        except Exception as e:
            print(f"\n[ERROR] Fatal error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            print("Engine shutdown complete")
    
    def _is_trading_allowed(self) -> tuple:
        """
        Check if trading is allowed based on weekly schedule.
        
        Returns:
            (allowed: bool, reason: str)
        """
        hours_config = self.config.get('trading_hours', {})
        
        if not hours_config.get('enabled', False):
            return True, "Trading hours not configured"
        
        # Get timezone
        tz_name = hours_config.get('timezone', 'America/Vancouver')
        try:
            tz = pytz.timezone(tz_name)
        except:
            tz = pytz.timezone('America/Vancouver')
        
        # Current time in configured timezone
        now = datetime.now(tz)
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # Parse schedule
        start = hours_config.get('week_start', {})
        end = hours_config.get('week_end', {})
        
        start_day = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 
                     'friday': 4, 'saturday': 5, 'sunday': 6}.get(start.get('day', 'sunday').lower(), 6)
        start_hour = start.get('hour', 14)
        start_minute = start.get('minute', 0)
        
        end_day = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                   'friday': 4, 'saturday': 5, 'sunday': 6}.get(end.get('day', 'friday').lower(), 4)
        end_hour = end.get('hour', 14)
        end_minute = end.get('minute', 0)
        
        # Convert to minutes from start of week (Sunday=0 in this system)
        # Adjust weekday: Python Monday=0, but we want Sunday=0
        adjusted_weekday = (weekday + 1) % 7  # Now Sunday=0, Monday=1, etc.
        current_mins = adjusted_weekday * 24 * 60 + now.hour * 60 + now.minute
        
        # Adjust start/end days to same system
        adjusted_start_day = (start_day + 1) % 7 if start_day != 6 else 0
        adjusted_end_day = (end_day + 1) % 7 if end_day != 6 else 0
        
        # For Sunday start, we need different logic
        # Sunday 2pm to Friday 2pm
        if start.get('day', '').lower() == 'sunday':
            adjusted_start_day = 0  # Sunday
        if end.get('day', '').lower() == 'friday':
            adjusted_end_day = 5  # Friday (0=Sun, 1=Mon, ..., 5=Fri)
        
        start_mins = adjusted_start_day * 24 * 60 + start_hour * 60 + start_minute
        end_mins = adjusted_end_day * 24 * 60 + end_hour * 60 + end_minute
        
        # Check if we're in trading window
        if start_mins <= current_mins <= end_mins:
            return True, f"Trading active ({now.strftime('%A %I:%M %p')} {tz_name})"
        else:
            if current_mins < start_mins:
                return False, f"Market opens Sunday 2:00 PM ({tz_name})"
            else:
                return False, f"Market closed Friday 2:00 PM ({tz_name})"
    
    def _check_week_end_close(self) -> bool:
        """Check if we should close all positions at week end."""
        hours_config = self.config.get('trading_hours', {})
        
        if not hours_config.get('enabled', False):
            return False
        
        if not hours_config.get('close_positions_at_week_end', False):
            return False
        
        tz_name = hours_config.get('timezone', 'America/Vancouver')
        try:
            tz = pytz.timezone(tz_name)
        except:
            tz = pytz.timezone('America/Vancouver')
        
        now = datetime.now(tz)
        end = hours_config.get('week_end', {})
        end_day = end.get('day', 'friday').lower()
        end_hour = end.get('hour', 14)
        end_minute = end.get('minute', 0)
        
        # Check if it's Friday and past close time
        if now.strftime('%A').lower() == end_day:
            close_time = now.replace(hour=end_hour, minute=end_minute, second=0)
            if now >= close_time:
                return True
        
        return False
    
    def _run_scan_cycle(self) -> None:
        """Execute one complete scan cycle."""
        cycle_start = datetime.now(timezone.utc)
        
        # Get local time for display
        hours_config = self.config.get('trading_hours', {})
        tz_name = hours_config.get('timezone', 'UTC')
        try:
            tz = pytz.timezone(tz_name)
            local_time = datetime.now(tz).strftime('%Y-%m-%d %I:%M:%S %p %Z')
        except:
            local_time = cycle_start.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        print(f"\n{'='*60}")
        print(f"SCAN CYCLE - {local_time}")
        print(f"{'='*60}")
        
        # Check trading hours
        trading_allowed, hours_reason = self._is_trading_allowed()
        print(f"Trading Hours: {hours_reason}")
        
        if not trading_allowed:
            print("⏸ Outside trading hours - waiting...")
            return
        
        # Check for Friday close
        if self._check_week_end_close():
            print("[STOP] WEEK END - Closing all positions...")
            positions = self.executor.get_open_positions()
            if positions:
                result = self.executor.close_all_positions()
                print(f"   Closed {result.get('closed', 0)} positions")
            print("⏸ Trading halted until Sunday 2:00 PM")
            return
        
        # Reset loop risk tracking
        self.risk_manager.reset_loop()
        
        # Get current session
        session = get_current_session()
        print(f"Session: {session}")
        if self.last_session != session:
            self.regime_state = {"metals_impulse": False}
            self.session_stats = {}
            self.last_session = session
        
        # Get account info
        account = self.executor.get_account_info()
        if not account:
            print("[!] Could not get account info")
            return
        
        print(f"Balance: ${account.get('balance', 0):,.2f} | Equity: ${account.get('equity', 0):,.2f}")

        # Sync MT5 deal history for per-family stats (live mode)
        if not self.config.get('execution', {}).get('dry_run', False):
            lookback = self.config.get('execution', {}).get('deal_history_lookback_minutes', 1440)
            sync = self.executor.sync_deal_history(lookback_minutes=lookback)
            if sync.get("new_deals", 0) > 0:
                print(f"[STATS] Deal history synced: {sync['new_deals']} new deals")
            family_stats = self.executor.get_family_stats()
            if family_stats:
                summary = self._format_family_stats(family_stats)
                if summary:
                    print(f"[STATS] By family: {summary}")

        # Regime detection: METALS_IMPULSE_PRIORITY_v1
        regimes_cfg = self.config.get("regimes", {}).get("metals_impulse", {})
        if regimes_cfg.get("enabled", True):
            now = datetime.now(timezone.utc)
            current_window = get_session_window(session, now)
            last_session = previous_session_name(session)
            last_window = get_session_window(last_session, now)
            groups = self.config.get("groups", {})
            current_deals = self.executor.get_deals_between(*current_window)
            last_deals = self.executor.get_deals_between(*last_window)
            session_stats = build_session_stats(current_deals, groups)
            last_session_stats = build_session_stats(last_deals, groups)
            metals_impulse = detect_metals_impulse(session_stats, last_session_stats)
            self.session_stats = {
                "current": session_stats,
                "last": last_session_stats,
                "session": session,
            }
            self.regime_state["metals_impulse"] = metals_impulse
            if metals_impulse:
                print("[REGIME] METALS_IMPULSE active")

        # Trade counter (training progress)
        counter = self.executor.get_trade_counter()
        if counter:
            target = counter.get("target_trades")
            target_text = f" / {target}" if target else ""
            print(f"[COUNTER] Trades since {counter.get('start_time')}: {counter.get('total_trades')}{target_text}")

        daily = self.executor.get_daily_trade_counter()
        if daily:
            print(f"[DAILY] Trades today ({daily.get('date')}): {daily.get('trades_today')}")

        # Daily RF progress log (live closed outcomes)
        if not self.config.get('execution', {}).get('dry_run', False):
            today = datetime.now(timezone.utc).date().isoformat()
            if self._last_rf_progress_date != today:
                target = int(self.config.get('execution', {}).get('rf_target_closed_trades', 1000))
                live_closed = self.executor.get_live_closed_outcome_count()
                print(f"[RF] Live closed outcomes: {live_closed} / {target}")
                self._last_rf_progress_date = today

        # Governance counter (independent of lifetime trades)
        governance = self.executor.get_governance_state()
        if governance:
            self._apply_governance_targets(governance)
            target = governance.get("target_trades", 1500)
            count = governance.get("governance_trade_count", 0)
            target_text = f"{count}/{target}" if target else f"{count}"
            min_score = self.governance_targets.get("min_score")
            min_conf = self.governance_targets.get("min_confidence")
            scale_status = "ENABLED" if self.governance_targets.get("scale_allowed") else "LOCKED"
            deploy_status = "READY" if self.governance_targets.get("deploy_ready") else "NO"
            if count < 300:
                min_text = "Ramping all families to 65 by 300"
            else:
                min_text = f"Min {min_score}/{min_conf}"
            print(
                f"[GOV] Governed trades: {target_text} | Phase: {self.governance_targets.get('phase')} "
                f"| {min_text} | Scale: {scale_status} | Deploy: {deploy_status}"
            )
            family_counts = governance.get("per_family_counts", {})
            if family_counts:
                summary = self._format_governance_counts(family_counts)
                if summary:
                    print(f"[GOV] By family: {summary}")
        
        # Check emergency conditions
        if self.risk_manager.should_emergency_close(account):
            print("🚨 EMERGENCY: Max drawdown exceeded!")
            result = self.executor.close_all_positions()
            print(f"   Closed {result.get('closed', 0)} positions")
            return
        
        # Get open positions
        positions = self.executor.get_open_positions()
        print(f"Open positions: {len(positions)}")
        
        # Monitor virtual positions in dry run mode
        if self.config.get('execution', {}).get('dry_run', False):
            closed_virtual = self.executor.monitor_virtual_positions()
            virtual_open = self.executor.get_virtual_positions()
            stats = self.executor.get_dry_run_stats()
            if virtual_open or stats['total_trades'] > 0:
                print(f"[DRY RUN] Virtual: {len(virtual_open)} open | "
                      f"{stats['wins']}W/{stats['losses']}L ({stats['win_rate']}%) | "
                      f"P&L: ${stats['total_profit']:+.2f}")
                family_stats = self.executor.get_family_stats()
                if family_stats:
                    summary = self._format_family_stats(family_stats)
                    if summary:
                        print(f"[DRY RUN] By family: {summary}")
        
        # Update profit locks on existing positions
        self._update_profit_locks(positions)
        
        # Get risk status
        risk_status = self.risk_manager.get_risk_status(positions, account)
        print(f"Risk Level: {risk_status['risk_level']} | Drawdown: {risk_status['drawdown']:.2f}%")
        
        # Scan each symbol
        signals = []
        
        for symbol, group in self.symbols.items():
            # Check session activation (per-symbol override supported)
            if not self._is_session_active(group, session, symbol=symbol):
                continue
            
            try:
                result = self._analyze_symbol(symbol, group, positions, account)
                if result:
                    signals.append(result)
            except Exception as e:
                print(f"  [!] {symbol}: Error - {str(e)[:50]}")
        
        # Sort by score and show top signals
        signals.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        if signals:
            print(f"\n[SIGNALS] Top Signals:")
            for sig in signals[:5]:
                direction = sig.get('direction', 'HOLD')
                dir_icon = "[+]" if direction == "BUY" else "[-]" if direction == "SELL" else "[=]"
                print(f"  {dir_icon} {sig['symbol']}: {direction} (Score: {sig['score']}, Conf: {sig.get('confidence', 0)}%)")
        
        print(f"\nCycle completed in {(datetime.now(timezone.utc) - cycle_start).total_seconds():.1f}s")
    
    def _analyze_symbol(
        self,
        symbol: str,
        group: str,
        positions: List[Dict[str, Any]],
        account: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single symbol and potentially trade."""
        
        # Load candle data
        df = self._load_candles(symbol)
        if df.empty or len(df) < 50:
            return None
        
        # === SCANNER MODE: Use Intelligent Scanner if available ===
        if self.scanner and self.config.get('scanner', {}).get('use_for_signals', True):
            return self._analyze_with_scanner(symbol, group, df, positions, account)
        
        # === STANDARD MODE: Use built-in indicators ===
        # Compute indicators
        indicators = compute_all_indicators(df)
        adx = indicators.get('adx14', 0)
        adx_min = self._get_group_adx_min(group)
        
        # Score the signal
        score_data = self.scorer.score_signal(indicators)
        score_data['adx_min'] = adx_min
        score = score_data['score']
        
        # Get group-specific threshold (looser for indices/metals/energy, tighter for FX/crypto)
        min_threshold = self._get_group_threshold(group)
        
        # Quick filter - skip scores well below threshold
        if score < min_threshold - 20:
            return None
        
        # Update tradeable flag based on group threshold
        score_data['tradeable'] = score >= min_threshold and score_data.get('direction') != 'HOLD'
        
        # Enforce ADX minimum per group
        if adx < adx_min:
            decision = {
                "direction": "HOLD",
                "confidence": 0,
                "sl": None,
                "tp": None,
                "risk_scale": 1.0,
                "reasoning": f"ADX {adx:.1f} below min {adx_min}"
            }
        # Enforce per-group RSI filter
        elif self._rsi_filtered(group, indicators):
            rsi = indicators.get('rsi14', 0)
            decision = {
                "direction": "HOLD",
                "confidence": 0,
                "sl": None,
                "tp": None,
                "risk_scale": 1.0,
                "reasoning": f"RSI {rsi:.1f} outside {group} bounds"
            }
        else:
            # Get AI decision or use score-based
            if self.ai_brain and score >= min_threshold - 10 and self.gpt_mode != "off":
                ai_cfg = self.config.get("ai", {})
                deploy_mode = ai_cfg.get("gpt_deploy_mode", "decide")
                ai_decision = self.ai_brain.make_decision(symbol, indicators, score_data)
                stage = self.gpt_mode

                if stage in ("observe", "shadow") or (stage == "deploy" and deploy_mode == "observe"):
                    label = "OBSERVE" if stage == "observe" else "SHADOW"
                    print(f"  [AI:{label}] {symbol}: {ai_decision.get('direction')} | {ai_decision.get('reasoning', '')}")
                    family_stats = self.executor.get_family_stats() if self.executor else {}
                    group_stats = family_stats.get(group, {})
                    wins = int(group_stats.get("wins", 0) or 0)
                    losses = int(group_stats.get("losses", 0) or 0)
                    win_rate = group_stats.get("win_rate", 0.0)
                    conf_min = self._get_group_confidence_min(group)
                    ai_conf = int(ai_decision.get("confidence", 0) or 0)
                    setup_quality = self._compute_setup_quality(score, ai_conf, conf_min)
                    regime_label = "METALS_IMPULSE" if self.regime_state.get("metals_impulse") else "NORMAL"
                    print(
                        f"    [AI:LOG] setup={setup_quality} | regime={regime_label} | "
                        f"family={group} {wins}W/{losses}L({win_rate}%) | "
                        f"conf_vs_outcome={ai_conf}%/{win_rate}%"
                    )
                    decision = self._score_based_decision(indicators, score_data)
                    self._log_shadow_decision(
                        symbol=symbol,
                        group=group,
                        gpt_decision=ai_decision,
                        rules_decision=decision,
                        score=score,
                        trade_placed=False,
                    )
                elif stage == "veto":
                    print(f"  [AI:VETO] {symbol}: {ai_decision.get('direction')} | {ai_decision.get('reasoning', '')}")
                    if ai_decision.get("direction", "HOLD").upper() == "HOLD":
                        decision = {
                            "direction": "HOLD",
                            "confidence": 0,
                            "sl": None,
                            "tp": None,
                            "risk_scale": 1.0,
                            "reasoning": "AI veto: HOLD"
                        }
                    else:
                        decision = self._score_based_decision(indicators, score_data)
                    veto_scale = float(ai_cfg.get("gpt_veto_risk_scale", 0.5))
                    decision["risk_scale"] = min(float(decision.get("risk_scale", 1.0)), veto_scale)
                elif stage == "scale":
                    print(f"  [AI:SCALE] {symbol}: {ai_decision.get('direction')} | {ai_decision.get('reasoning', '')}")
                    decision = ai_decision
                    scale_cap = ai_cfg.get("gpt_scale_max_risk_scale")
                    if isinstance(scale_cap, (int, float)):
                        decision["risk_scale"] = min(float(decision.get("risk_scale", 1.0)), float(scale_cap))
                else:
                    decision = ai_decision
            else:
                # Score-based decision (no AI)
                decision = self._score_based_decision(indicators, score_data)
        
        direction = decision.get('direction', 'HOLD')
        confidence = decision.get('confidence', 0)
        conf_min = self._get_group_confidence_min(group)

        # Regime rules: METALS_IMPULSE_PRIORITY_v1
        if self.regime_state.get("metals_impulse") and group.lower() == "crypto":
            confidence = max(0, int(confidence - 10))
            decision["confidence"] = confidence
        
        if direction != "HOLD" and confidence < conf_min:
            decision = {
                "direction": "HOLD",
                "confidence": 0,
                "sl": None,
                "tp": None,
                "risk_scale": 1.0,
                "reasoning": f"Confidence {confidence}% below min {conf_min}%"
            }
            direction = "HOLD"
            confidence = 0
        
        # Log analysis
        status = "[OK]" if direction != "HOLD" else "[--]"
        print(f"  {status} {symbol} [{group}]: Score={score}, Dir={direction}, Conf={confidence}%")

        # ============================================
        # INTELLIGENT SCANNER CONFIDENCE GATE
        # ============================================
        if direction != "HOLD" and self.intelligent_scanner_enabled and self.intelligent_scanner:
            try:
                scan_result = self.intelligent_scanner.scan_symbol(symbol)
                if scan_result and scan_result.get("final_confidence") is not None:
                    scanner_conf = int(scan_result["final_confidence"])
                    scanner_dir = scan_result.get("direction")
                    session = get_current_session()
                    min_conf = self._get_scanner_session_confidence(session)

                    if scanner_conf < min_conf:
                        print(f"    [SCANNER:BLOCKED] {symbol}: confidence {scanner_conf}% < {min_conf}% (session={session})")
                        direction = "HOLD"
                        confidence = 0
                    else:
                        engine_dir = direction.upper()
                        s_dir = (scanner_dir or "").upper()
                        if s_dir in ("LONG", "BUY"):
                            s_dir = "BUY"
                        elif s_dir in ("SHORT", "SELL"):
                            s_dir = "SELL"
                        if s_dir and s_dir != engine_dir and s_dir != "HOLD":
                            decision["risk_scale"] = decision.get("risk_scale", 1.0) * 0.5
                            print(f"    [SCANNER:WARN] {symbol}: direction mismatch (engine={engine_dir}, scanner={s_dir}) conf={scanner_conf}% — half risk")
                        else:
                            print(f"    [SCANNER:OK] {symbol}: confidence={scanner_conf}% (min={min_conf}%, session={session})")
                else:
                    print(f"    [SCANNER] {symbol}: no result — passing on score only")
            except Exception as e:
                print(f"    [SCANNER] {symbol}: error — {e}")

        # ============================================
        # R:R BLOCKER (per symbol + session)
        # ============================================
        if direction != "HOLD" and self.rr_blocker_enabled:
            session = get_current_session()
            blocked, reason = self._check_rr_blocker(symbol, session)
            if blocked:
                print(f"    [RR:BLOCKED] {symbol} ({session}): {reason}")
                direction = "HOLD"
                confidence = 0

        # Hard daily trade cap (uses persisted daily counter)
        if direction != "HOLD":
            max_daily = int(self.config.get("risk", {}).get("max_daily_trades", 20))
            daily = self.executor.get_daily_trade_counter() if self.executor else {}
            trades_today = int(daily.get("trades_today", 0) or 0)
            if trades_today >= max_daily:
                print(f"    [BLOCKED] Daily trade cap reached ({trades_today}/{max_daily})")
                direction = "HOLD"
                confidence = 0

        # Determine decision source based on GPT stage
        if self.gpt_mode in ("observe", "shadow"):
            _dsrc = "RULES"
        elif self.gpt_mode == "veto":
            _dsrc = "GPT_ASSIST"
        elif self.gpt_mode == "scale":
            _dsrc = "GPT_ASSIST"
        elif self.gpt_mode == "deploy":
            _dsrc = "GPT_PRIMARY"
        else:
            _dsrc = "RULES"

        # Execute if tradeable
        if direction != "HOLD" and score_data.get('tradeable', False):
            self._attempt_trade(
                symbol, group, direction, decision,
                positions, account, indicators,
                score_data=score_data,
                decision_source=_dsrc,
            )
        
        return {
            "symbol": symbol,
            "group": group,
            "score": score,
            "direction": direction,
            "confidence": confidence,
            "risk_level": indicators.get('risk_level', 'NORMAL')
        }
    
    def _attempt_trade(
        self,
        symbol: str,
        group: str,
        direction: str,
        decision: Dict[str, Any],
        positions: List[Dict[str, Any]],
        account: Dict[str, Any],
        indicators: Dict[str, Any],
        score_data: Optional[Dict[str, Any]] = None,
        decision_source: str = "RULES",
    ) -> None:
        """Attempt to place a trade after all checks."""
        
        # Get risk-adjusted scale
        base_scale = decision.get('risk_scale', 1.0)
        if not self.governance_targets.get("scale_allowed", True):
            base_scale = min(1.0, base_scale)
        max_scale = self.governance_targets.get("max_risk_scale")
        if isinstance(max_scale, (int, float)):
            base_scale = min(float(max_scale), base_scale)
        risk_scale = self.risk_manager.get_adjusted_risk_scale(base_scale, account, group)

        trade_tags = {
            "decision_source": decision_source,
            "rules_version_id": self.rules_version_id,
        }
        if self.regime_state.get("metals_impulse"):
            trade_tags.update({
                "regime": "METALS_IMPULSE",
                "priority": True,
                "session_origin": "CONTINUATION"
            })
            if group.lower() == "metals":
                decision["allow_runner_extension"] = True
                decision["tighten_profit_lock"] = False
                decision["cap_winners"] = False
            if group.lower() == "crypto":
                crypto_losses = (
                    self.session_stats.get("current", {})
                    .get("crypto", {})
                    .get("losses", 0)
                )
                if crypto_losses >= 1:
                    print("    [BLOCKED] METALS_IMPULSE crypto loss cap reached for session")
                    return
        
        # Check if trade is allowed
        check = self.risk_manager.can_trade(symbol, group, positions, account, risk_scale)
        
        if not check['allowed']:
            print(f"    [BLOCKED] {check['reason']}")
            return
        
        # Execute trade
        sl = decision.get('sl')
        tp = decision.get('tp')
        
        result = self.executor.execute_trade(
            symbol=symbol,
            direction=direction,
            sl=sl,
            tp=tp,
            risk_scale=risk_scale,
            comment=f"REBEL_{group.upper()}",
            tags=trade_tags
        )
        
        if result.get('success'):
            self.risk_manager.register_trade(risk_scale)
            print(f"    [TRADE] Trade placed: {direction} @ {result.get('price', 0):.5f}")
            print(f"       SL: {f'{sl:.5f}' if sl else 'None'} | TP: {f'{tp:.5f}' if tp else 'None'}")
            self._log_rf_features(
                ticket=result.get('ticket'),
                symbol=symbol,
                group=group,
                direction=direction,
                indicators=indicators,
                score_data=score_data or {},
                decision=decision,
                decision_source=decision_source,
            )
        else:
            print(f"    [FAILED] Trade failed: {result.get('error', 'Unknown')}")
    
    def _analyze_with_scanner(
        self,
        symbol: str,
        group: str,
        df: pd.DataFrame,
        positions: List[Dict[str, Any]],
        account: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze symbol using the Intelligent Scanner.
        Integrates scanner signals with the rules-based engine.
        """
        # Convert DataFrame to candle list for scanner
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "tick_volume": int(row.get('tick_volume', row.get('volume', 0)))
            })
        
        # Use scanner to analyze
        use_aim = self.config.get('scanner', {}).get('use_aim', True)
        scan_result = self.scanner.scan_symbol(symbol, candles, use_aim=use_aim)
        
        if not scan_result:
            return None
        
        # Extract scanner values
        direction = scan_result.get('direction', 'HOLD')
        confidence = scan_result.get('confidence', 0)
        safety = scan_result.get('safety', 'UNKNOWN')
        risk = scan_result.get('risk', 'MEDIUM')
        conf_min = self._get_group_confidence_min(group)

        # Regime rules: METALS_IMPULSE_PRIORITY_v1
        if self.regime_state.get("metals_impulse") and group.lower() == "crypto":
            confidence = max(0, int(confidence - 10))
        
        # Skip unsafe signals or low confidence
        if safety != 'SAFE' or confidence < conf_min:
            return None
        
        # Log analysis
        status = "[OK]" if direction != "HOLD" else "[--]"
        sparkline = scan_result.get('sparkline', '')
        print(f"  {status} {symbol} [{group}]: {direction} ({confidence}%) {sparkline}")
        if scan_result.get('ai_comment'):
            print(f"      [AI] {scan_result['ai_comment'][:60]}")
        
        # Build decision from scanner result
        indicators = scan_result.get('indicators', {})
        atr = indicators.get('atr', 0)
        close = indicators.get('current_price', 0)
        adx = indicators.get('adx', 0)
        adx_min = self._get_group_adx_min(group)

        if adx and adx < adx_min:
            print(f"  [--] {symbol} [{group}]: ADX {adx:.1f} < {adx_min} (min)")
            return None
        
        # Calculate SL/TP from scanner's ATR
        if direction == "BUY" and atr > 0:
            sl = close - (1.0 * atr)
            tp = close + (2.0 * atr)
        elif direction == "SELL" and atr > 0:
            sl = close + (1.0 * atr)
            tp = close - (2.0 * atr)
        else:
            sl = None
            tp = None
        
        decision = {
            "direction": direction,
            "confidence": confidence,
            "sl": sl,
            "tp": tp,
            "risk_scale": 1.0 if risk == "LOW" else (0.8 if risk == "MEDIUM" else 0.5),
            "reasoning": f"Scanner: {scan_result.get('trend', 'N/A')} trend, {scan_result.get('pattern', 'none')}"
        }
        
        # Get group-specific threshold
        min_threshold = self._get_group_threshold(group)
        
        # Build score_data equivalent for compatibility
        score_data = {
            "score": confidence,
            "direction": direction,
            "confidence": confidence,
            "tradeable": direction != "HOLD" and confidence >= min_threshold and safety == "SAFE"
        }
        
        # Execute if tradeable
        if score_data.get('tradeable', False):
            self._attempt_trade(
                symbol, group, direction, decision,
                positions, account, indicators,
                score_data=score_data,
                decision_source="RULES",
            )
        
        return {
            "symbol": symbol,
            "group": group,
            "score": confidence,
            "direction": direction,
            "confidence": confidence,
            "risk_level": risk,
            "source": "scanner"
        }
    
    def _score_based_decision(
        self,
        indicators: Dict[str, Any],
        score_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make decision based purely on score (no AI).
        Used when AI is disabled or unavailable.
        """
        direction = score_data.get('direction', 'HOLD')
        score = score_data.get('score', 0)
        
        if direction == "HOLD" or score < 75:
            return {
                "direction": "HOLD",
                "confidence": 0,
                "sl": None,
                "tp": None,
                "risk_scale": 1.0,
                "reasoning": "Score too low or no clear direction"
            }
        
        # Calculate SL/TP based on ATR
        close = indicators.get('close', 0)
        atr = indicators.get('atr14', 0)
        
        if direction == "BUY":
            sl = close - (1.0 * atr) if atr > 0 else None
            tp = close + (2.0 * atr) if atr > 0 else None
        else:
            sl = close + (1.0 * atr) if atr > 0 else None
            tp = close - (2.0 * atr) if atr > 0 else None
        
        # Risk scale based on score
        if score >= 80:
            risk_scale = 1.3
        elif score >= 70:
            risk_scale = 1.1
        else:
            risk_scale = 1.0
        
        return {
            "direction": direction,
            "confidence": score_data.get('confidence', score),
            "sl": sl,
            "tp": tp,
            "risk_scale": risk_scale,
            "reasoning": f"Score-based: {score}/100"
        }
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get the correct pip/tick size for a symbol."""
        s = symbol.upper()
        try:
            info = mt5.symbol_info(s)
            if info and info.trade_tick_size > 0:
                return info.trade_tick_size
        except Exception:
            pass
        # Fallback approximations
        if any(m in s for m in ("XAU",)):
            return 0.01
        if any(m in s for m in ("XAG",)):
            return 0.001
        if any(m in s for m in ("XPT", "XPD")):
            return 0.01
        if any(m in s for m in ("BTC",)):
            return 1.0
        if any(m in s for m in ("ETH",)):
            return 0.01
        if any(m in s for m in ("US500", "US30", "USTECH", "US2000", "NAS", "DAX", "GER", "UK100", "AUS200", "JPN225")):
            return 0.1
        if "JPY" in s:
            return 0.01 if "-" not in s else 0.001
        return 0.0001

    def _update_profit_locks(self, positions: List[Dict[str, Any]]) -> None:
        """Update profit locks on open positions."""
        if not self.config.get('profit_lock', {}).get('enabled', True):
            return
        
        for pos in positions:
            try:
                symbol = pos['symbol']
                current = pos.get('price_current', 0)
                
                if current <= 0:
                    continue
                
                pip_value = self._get_pip_value(symbol)
                
                new_sl = self.profit_lock.get_new_sl(pos, current, pip_value)
                
                if new_sl:
                    # Round to symbol's digit precision
                    try:
                        info = mt5.symbol_info(symbol)
                        if info:
                            new_sl = round(new_sl, info.digits)
                    except Exception:
                        pass
                    result = self.executor.modify_position(pos['ticket'], sl=new_sl)
                    if result.get('success'):
                        print(f"  [LOCK] Profit locked on {symbol}: SL moved to {new_sl:.5f}")
                    else:
                        print(f"  [LOCK] Failed to lock {symbol}: {result.get('error', 'unknown')}")
                        
            except Exception as e:
                print(f"  [LOCK] Error on {pos.get('symbol', '?')}: {e}")
    
    def _init_mt5_with_retry(self, max_retries: int = 5, retry_delay: int = 3) -> bool:
        """
        Initialize MT5 with retry logic.
        Handles 'resetting trading platform' state.
        """
        for attempt in range(1, max_retries + 1):
            try:
                # Shutdown any existing connection
                try:
                    mt5.shutdown()
                except:
                    pass
                
                time.sleep(1)
                
                # Get path from config
                mt5_config = self.config.get('mt5', {})
                path = mt5_config.get('path')
                
                if path:
                    if not mt5.initialize(path):
                        print(f"[ENGINE] MT5 init failed (attempt {attempt}/{max_retries})")
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                        continue
                else:
                    if not mt5.initialize():
                        print(f"[ENGINE] MT5 init failed (attempt {attempt}/{max_retries})")
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                        continue
                
                # Wait for terminal to load
                time.sleep(2)
                
                # Verify terminal info
                terminal_ready = False
                for check in range(3):
                    if mt5.terminal_info() is not None:
                        terminal_ready = True
                        break
                    time.sleep(2)
                
                if not terminal_ready:
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                    continue
                
                # Verify account
                if mt5.account_info() is not None:
                    return True
                    
            except Exception as e:
                print(f"[ENGINE] MT5 error (attempt {attempt}): {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
        
        return False
    
    def _load_candles(self, symbol: str, count: int = 250) -> pd.DataFrame:
        """Load candle data from MT5."""
        if not self._init_mt5_with_retry():
            return pd.DataFrame()
        
        try:
            timeframe = self._get_timeframe()
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            df.columns = [c.lower() for c in df.columns]
            
            return df
            
        except Exception:
            return pd.DataFrame()
        finally:
            mt5.shutdown()
    
    def _get_timeframe(self) -> int:
        """Get MT5 timeframe constant."""
        tf = self.config.get('timeframe', 'M15')
        
        timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        
        return timeframes.get(tf, mt5.TIMEFRAME_M15)
    
    def _is_session_active(self, group: str, session: str, symbol: str = "") -> bool:
        """Check if trading is allowed for this group/symbol in current session."""
        # Per-symbol override takes priority
        if symbol:
            symbol_sessions = self.config.get('symbol_session_overrides', {})
            override = symbol_sessions.get(symbol.upper())
            if override:
                return session.upper() in [s.upper() for s in override]

        session_map = self.config.get('session_activation', {})
        allowed = session_map.get(group, ['TOKYO', 'LONDON', 'NEW_YORK'])
        
        return session.upper() in [s.upper() for s in allowed]
    
    def _get_group_threshold(self, group: str) -> int:
        """
        Get the minimum score threshold for a group.
        Allows looser filters for trending assets (indices, metals, energy)
        and tighter filters for noisy assets (FX, crypto).
        """
        scoring_config = self.config.get('scoring', {})
        group_thresholds = scoring_config.get('group_thresholds', {})
        default_threshold = scoring_config.get('min_score_to_trade', 60)
        
        base = group_thresholds.get(group, default_threshold)
        stage_min = self._get_governed_min_for_group(group, base)
        return max(base, stage_min) if isinstance(stage_min, int) else base

    def _get_group_adx_min(self, group: str) -> int:
        """Get the minimum ADX required for a group."""
        scoring_config = self.config.get('scoring', {})
        group_adx = scoring_config.get('adx_min_by_group', {})
        default_adx = scoring_config.get('adx_min', 20)
        return int(group_adx.get(group, default_adx))

    def _rsi_filtered(self, group: str, indicators: Dict[str, Any]) -> bool:
        """Check if RSI is outside the allowed range for this group. Returns True if filtered out."""
        scoring_config = self.config.get('scoring', {})
        rsi_filters = scoring_config.get('rsi_filter_by_group', {})
        group_filter = rsi_filters.get(group)
        if not group_filter:
            return False
        rsi = indicators.get('rsi14', 50)
        direction = indicators.get('trend_direction', '')
        long_min = group_filter.get('long_min', 0)
        short_max = group_filter.get('short_max', 100)
        if direction in ('BUY', 'BULLISH', 'UP') and rsi < long_min:
            return True
        if direction in ('SELL', 'BEARISH', 'DOWN') and rsi > short_max:
            return True
        return False

    def _get_group_confidence_min(self, group: str) -> int:
        """Get the minimum confidence required for a group."""
        scoring_config = self.config.get('scoring', {})
        group_conf = scoring_config.get('confidence_thresholds', {})
        default_conf = scoring_config.get('min_confidence_to_trade', 60)
        base = int(group_conf.get(group, default_conf))
        governed_min = self.governance_targets.get("min_confidence")
        return max(base, int(governed_min)) if isinstance(governed_min, int) else base

    def _get_scanner_session_confidence(self, session: str) -> int:
        """Get minimum scanner confidence for the current session."""
        return int(self.scanner_session_confidence.get(
            session.upper(), self.scanner_global_min_conf
        ))

    # ------------------------------------------------------------------
    # R:R BLOCKER — blocks symbol+session combos with negative cumulative
    # R:R after a minimum number of trades
    # ------------------------------------------------------------------

    def _check_rr_blocker(self, symbol: str, session: str) -> tuple:
        """
        Returns (blocked: bool, reason: str).
        Rebuilds the lookup cache once per scan cycle (every 60s max).
        """
        now = datetime.now(timezone.utc)
        if self._rr_blocker_cache_ts is None or (now - self._rr_blocker_cache_ts).total_seconds() > 120:
            self._rr_blocker_cache = self._build_rr_blocker_cache()
            self._rr_blocker_cache_ts = now

        key = f"{symbol.upper()}|{session.upper()}"
        entry = self._rr_blocker_cache.get(key)
        if entry is None:
            return False, ""
        trades = entry["trades"]
        rr = entry["rr"]
        if trades >= self.rr_blocker_min_trades and rr < 0:
            return True, f"cumulative R:R={rr:+.2f} over {trades} trades"
        return False, ""

    def _build_rr_blocker_cache(self) -> Dict[str, Any]:
        """
        Scan trades.jsonl for CLOSED outcomes and accumulate wins/losses
        per symbol+session.  Uses the trade's timestamp to derive session.
        """
        cache: Dict[str, dict] = {}
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "trades.jsonl"
        )
        if not os.path.exists(log_path):
            return cache
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("status") != "CLOSED" and rec.get("direction") != "CLOSE":
                        continue
                    sym = (rec.get("symbol") or "").upper()
                    profit = rec.get("profit")
                    if profit is None:
                        continue
                    profit = float(profit)
                    ts_str = rec.get("timestamp")
                    session = self._derive_session_from_timestamp(ts_str)
                    if not session:
                        continue
                    key = f"{sym}|{session}"
                    if key not in cache:
                        cache[key] = {"trades": 0, "total_win": 0.0, "total_loss": 0.0, "rr": 0.0}
                    bucket = cache[key]
                    bucket["trades"] += 1
                    if profit > 0:
                        bucket["total_win"] += profit
                    else:
                        bucket["total_loss"] += abs(profit)
            for bucket in cache.values():
                tw = bucket["total_win"]
                tl = bucket["total_loss"]
                bucket["rr"] = (tw / tl) if tl > 0 else (999.0 if tw > 0 else 0.0)
                bucket["rr"] = round(bucket["rr"] - 1.0, 4)
        except Exception:
            pass
        return cache

    @staticmethod
    def _derive_session_from_timestamp(ts_str: str) -> str:
        if not ts_str:
            return ""
        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            hour = dt.astimezone(timezone.utc).hour
            weekday = dt.astimezone(timezone.utc).weekday()
            if weekday >= 5:
                return "WEEKEND"
            if 7 <= hour < 13:
                return "LONDON"
            elif 13 <= hour < 21:
                return "NEW_YORK"
            else:
                return "TOKYO"
        except Exception:
            return ""

    def _compute_governance_targets(self, count: int) -> Dict[str, Any]:
        """Compute governed training targets based on trade count."""
        if count < 500:
            min_score = 55 if count >= 300 else 50
            return {
                "phase": "PHASE_1",
                "min_score": min_score,
                "min_confidence": 75,
                "scale_allowed": False,
                "deploy_ready": False,
                "max_risk_scale": 1.0,
            }
        if count < 750:
            return {
                "phase": "PHASE_2",
                "min_score": 60,
                "min_confidence": 75,
                "scale_allowed": False,
                "deploy_ready": False,
                "max_risk_scale": 1.0,
            }
        if count < 1000:
            return {
                "phase": "PHASE_3",
                "min_score": 65,
                "min_confidence": 75,
                "scale_allowed": True,
                "deploy_ready": False,
                "max_risk_scale": 1.1,
            }
        if count < 1500:
            return {
                "phase": "PHASE_4",
                "min_score": 70,
                "min_confidence": 75,
                "scale_allowed": True,
                "deploy_ready": False,
                "max_risk_scale": 1.2,
            }
        return {
            "phase": "DEPLOY_READY",
            "min_score": 70,
            "min_confidence": 75,
            "scale_allowed": True,
            "deploy_ready": True,
            "max_risk_scale": 1.3,
        }

    def _apply_governance_targets(self, governance_state: Dict[str, Any]) -> None:
        """Update governance targets using current governed trade count."""
        self.governance_state = dict(governance_state)
        count = int(self.governance_state.get("governance_trade_count", 0) or 0)
        self.governance_targets = self._compute_governance_targets(count)
        self.gpt_mode = self._compute_gpt_mode(count)

    def _compute_gpt_mode(self, count: int) -> str:
        """Compute GPT staging mode based on governed trade count."""
        ai_cfg = self.config.get("ai", {})
        if not ai_cfg.get("enabled", True) or not ai_cfg.get("gpt_enabled", True):
            return "off"
        if not self.ai_brain:
            return "off"
        observe_at = int(ai_cfg.get("gpt_observe_at", 0))
        shadow_start = int(ai_cfg.get("gpt_shadow_start", ai_cfg.get("gpt_decide_start", 1000)))
        veto_start = int(ai_cfg.get("gpt_veto_start", ai_cfg.get("gpt_decide_end", 1250)))
        scale_start = int(ai_cfg.get("gpt_scale_start", 1500))
        deploy_start = int(ai_cfg.get("gpt_deploy_start", 1750))

        if count < observe_at:
            return "off"
        if count < shadow_start:
            return "observe"
        if count < veto_start:
            return "shadow"
        if count < scale_start:
            return "veto"
        if count < deploy_start:
            return "scale"
        return "deploy"

    def _get_governed_min_for_group(self, group: str, base: int) -> Optional[int]:
        """Return governed min threshold for a group based on trade count."""
        count = int(self.governance_state.get("governance_trade_count", 0) or 0)
        if count < 300:
            target = 55
            if base >= target:
                return base
            progress = max(0.0, min(1.0, count / 300))
            return int(round(base + (target - base) * progress))
        if count < 500:
            return 55
        if count < 750:
            return 60
        if count < 1000:
            return 65
        return 70

    def _format_governance_counts(self, counts: Dict[str, Any]) -> str:
        """Format governed trade counts by family."""
        order = ["majors", "crosses", "exotics", "metals", "indices", "energy", "softs", "crypto", "unknown"]
        parts = []
        for key in order:
            if key not in counts:
                continue
            parts.append(f"{key}:{counts.get(key, 0)}")
        return " | ".join(parts)

    def _format_family_stats(self, family_stats: Dict[str, Any]) -> str:
        """Format per-family stats for logging."""
        order = ["majors", "crosses", "metals", "indices", "energy", "softs", "crypto", "unknown"]
        parts = []
        for key in order:
            if key not in family_stats:
                continue
            stats = family_stats[key]
            if key == "unknown" and len(family_stats) > 1:
                continue
            parts.append(f"{key}:{stats.get('wins',0)}W/{stats.get('losses',0)}L({stats.get('win_rate',0)}%)")
        return " | ".join(parts)

    def _compute_setup_quality(self, score: int, confidence: int, conf_min: int) -> str:
        """Grade setup quality for observe-only logging."""
        if score >= 80 and confidence >= conf_min + 10:
            return "A"
        if score >= 75 and confidence >= conf_min:
            return "B"
        if score >= 65:
            return "C"
        return "D"
    
    def _log_shadow_decision(
        self,
        symbol: str,
        group: str,
        gpt_decision: Dict[str, Any],
        rules_decision: Dict[str, Any],
        score: int,
        trade_placed: bool,
    ) -> None:
        """Persist a GPT shadow decision to gpt_shadow_log.jsonl for later analysis."""
        try:
            gov = self.governance_state or {}
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "rules_version_id": self.rules_version_id,
                "governed_count": int(gov.get("governance_trade_count", 0) or 0),
                "symbol": symbol,
                "group": group,
                "score": score,
                "gpt_direction": gpt_decision.get("direction", "HOLD"),
                "gpt_confidence": gpt_decision.get("confidence", 0),
                "gpt_risk_scale": gpt_decision.get("risk_scale", 1.0),
                "gpt_sl_atr": gpt_decision.get("sl_atr_multiplier", 1.0),
                "gpt_tp_atr": gpt_decision.get("tp_atr_multiplier", 2.0),
                "gpt_reasoning": (gpt_decision.get("reasoning") or "")[:200],
                "rules_direction": rules_decision.get("direction", "HOLD"),
                "rules_confidence": rules_decision.get("confidence", 0),
                "rules_risk_scale": rules_decision.get("risk_scale", 1.0),
                "agree": (
                    gpt_decision.get("direction", "HOLD") == rules_decision.get("direction", "HOLD")
                ),
                "trade_placed": trade_placed,
            }
            os.makedirs(os.path.dirname(self._shadow_log_path), exist_ok=True)
            with open(self._shadow_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"  [SHADOW LOG] Write error: {e}")

    # --- RF Feature columns (order matters — keep in sync with header) ---
    RF_COLUMNS = [
        "timestamp", "ticket", "symbol", "group", "direction",
        "decision_source", "rules_version_id", "governed_count",
        "session", "close", "open", "high", "low", "volume",
        "ema9", "ema21", "ema50", "sma20", "sma50",
        "rsi14", "rsi7", "adx14", "atr14", "atr_percent",
        "macd_line", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
        "trend", "trend_strength", "volatility", "momentum", "risk_level",
        "structure", "higher_highs", "lower_lows",
        "score", "confidence",
        "score_trend", "score_momentum", "score_pattern", "score_structure",
    ]

    def _log_rf_features(
        self,
        ticket: Any,
        symbol: str,
        group: str,
        direction: str,
        indicators: Dict[str, Any],
        score_data: Dict[str, Any],
        decision: Dict[str, Any],
        decision_source: str = "RULES",
    ) -> None:
        """Write one feature row to rf_features.csv at trade entry time."""
        try:
            gov = self.governance_state or {}
            mkt = indicators.get("market_structure", {})
            comps = score_data.get("components", {})
            session = get_current_session()

            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticket": ticket,
                "symbol": symbol,
                "group": group,
                "direction": direction,
                "decision_source": decision_source,
                "rules_version_id": self.rules_version_id,
                "governed_count": int(gov.get("governance_trade_count", 0) or 0),
                "session": session,
                "close": indicators.get("close", 0),
                "open": indicators.get("open", 0),
                "high": indicators.get("high", 0),
                "low": indicators.get("low", 0),
                "volume": indicators.get("volume", 0),
                "ema9": indicators.get("ema9", 0),
                "ema21": indicators.get("ema21", 0),
                "ema50": indicators.get("ema50", 0),
                "sma20": indicators.get("sma20", 0),
                "sma50": indicators.get("sma50", 0),
                "rsi14": indicators.get("rsi14", 0),
                "rsi7": indicators.get("rsi7", 0),
                "adx14": indicators.get("adx14", 0),
                "atr14": indicators.get("atr14", 0),
                "atr_percent": indicators.get("atr_percent", 0),
                "macd_line": indicators.get("macd_line", 0),
                "macd_signal": indicators.get("macd_signal", 0),
                "macd_histogram": indicators.get("macd_histogram", 0),
                "bb_upper": indicators.get("bb_upper", 0),
                "bb_middle": indicators.get("bb_middle", 0),
                "bb_lower": indicators.get("bb_lower", 0),
                "bb_width": indicators.get("bb_width", 0),
                "bb_position": indicators.get("bb_position", 0),
                "trend": indicators.get("trend", ""),
                "trend_strength": indicators.get("trend_strength", ""),
                "volatility": indicators.get("volatility", ""),
                "momentum": indicators.get("momentum", ""),
                "risk_level": indicators.get("risk_level", ""),
                "structure": mkt.get("structure", ""),
                "higher_highs": mkt.get("higher_highs", False),
                "lower_lows": mkt.get("lower_lows", False),
                "score": score_data.get("score", 0),
                "confidence": score_data.get("confidence", 0),
                "score_trend": comps.get("trend", {}).get("score", 0),
                "score_momentum": comps.get("momentum", {}).get("score", 0),
                "score_pattern": comps.get("pattern", {}).get("score", 0),
                "score_structure": comps.get("structure", {}).get("score", 0),
            }

            os.makedirs(os.path.dirname(self._rf_features_path), exist_ok=True)
            write_header = not os.path.exists(self._rf_features_path) or \
                           os.path.getsize(self._rf_features_path) == 0
            with open(self._rf_features_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.RF_COLUMNS)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"    [RF] Feature logged for ticket {ticket} ({symbol} {direction})")
        except Exception as e:
            print(f"    [RF] Feature log error: {e}")

    def _build_symbol_list(self) -> Dict[str, str]:
        """Build symbol -> group mapping."""
        symbols = {}
        groups = self.config.get('groups', {})
        enabled = self.config.get('enabled_groups', list(groups.keys()))
        suffix = self.config.get('symbol_suffix', '')
        
        for group_name, group_symbols in groups.items():
            if group_name in enabled:
                for sym in group_symbols:
                    full_symbol = f"{sym}{suffix}"
                    symbols[full_symbol] = group_name
        
        return symbols
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[!] Config not found: {config_path}")
            print("  Using default configuration")
            return self._default_config()
        except Exception as e:
            print(f"[!] Config error: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "scan_interval": 60,
            "timeframe": "M15",
            "symbol_suffix": "",
            
            "groups": {
                "majors": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                "metals": ["XAUUSD"],
                "indices": ["US500", "NAS100"]
            },
            
            "enabled_groups": ["majors", "metals", "indices"],
            
            "session_activation": {
                "majors": ["TOKYO", "LONDON", "NEW_YORK"],
                "metals": ["TOKYO", "LONDON", "NEW_YORK"],
                "indices": ["LONDON", "NEW_YORK"]
            },
            
            "ai": {
                "enabled": True,
                "model": "gpt-4o",
                "temperature": 0.1
            },
            
            "scoring": {
                "min_score_to_trade": 60
            },
            
            "risk": {
                "max_total_trades": 6,
                "max_drawdown_percent": 6.0,
                "max_daily_trades": 20,
                "max_risk_per_loop": 2.0,
                "group_limits": {
                    "majors": {"max_trades": 2, "risk_scale": 1.0},
                    "metals": {"max_trades": 1, "risk_scale": 1.0},
                    "indices": {"max_trades": 1, "risk_scale": 0.8}
                }
            },
            
            "execution": {
                "dry_run": False,
                "risk_per_trade": 1.0,
                "log_dir": "logs"
            },
            
            "profit_lock": {
                "enabled": True,
                "profit_ladder": [
                    {"trigger": 10, "lock": 0},
                    {"trigger": 20, "lock": 10},
                    {"trigger": 40, "lock": 25},
                    {"trigger": 60, "lock": 45}
                ]
            }
        }


def run_engine(config_path: str = "config/rebel_config.yaml") -> None:
    """Entry point to run the engine."""
    engine = RebelEngine(config_path)
    engine.run()

