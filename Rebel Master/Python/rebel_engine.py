import MetaTrader5 as mt5
import yaml
import json
import os
import sys
from typing import Any, Dict
from datetime import datetime, timedelta

import pandas as pd

try:
    import openai as _openai
    openai = _openai
    HAS_OPENAI = True
except ImportError:
    openai = None
    HAS_OPENAI = False

from ml_feature_logger import MLFeatureLogger
from ml_label_generator import MLLabelGenerator
from rebel_position_manager import RebelPositionManager
from rebel_profit_lock import ProfitLockManager
from governance.regime_guard import observe_metals_impulse
from rebel_signal_filters import run_all_filters
from rebel_trade_limiter import (
    can_trade as limiter_can_trade,
    record_trade_opened as limiter_record_trade,
    record_filter_passed as limiter_record_filter_passed,
)

# Intelligent Scanner imports
sys.path.insert(0, r"C:\Rebel Technologies\Rebel Master\Scanner")
try:
    from rebel_intelligent_scanner import RebelIntelligentScanner
    HAS_INTELLIGENT_SCANNER = True
except ImportError as e:
    print(f"[WARN] Intelligent Scanner not available: {e}")
    HAS_INTELLIGENT_SCANNER = False

# ML Shadow Mode imports
sys.path.insert(0, r"C:\Rebel Technologies\Rebel Master\ML")
try:
    from ml_trade_filter import MLTradeFilter, MLFilterConfig, shadow_mode_decision
    from ml_trade_logger import log_ml_decision
    ML_AVAILABLE = True
    print("[ML] ML Trade Filter module loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"[ML] ML Trade Filter not available: {e}")

# ML Model paths
ML_MODEL_PATHS = {
    "rf_v1": r"C:\Rebel Technologies\Rebel Master\ML\model_rf_v1.joblib",
    "rf_v2": r"C:\Rebel Technologies\Rebel Master\ML\model_rf_v2.joblib",
    "rf_v3": r"C:\Rebel Technologies\Rebel Master\ML\model_rf_v3.joblib",  # v3 Beta - optimized for small datasets
    "rf_v3_production": r"C:\Rebel Technologies\Rebel Master\ML\model_rf_v3_production.joblib",  # v3 Production - 500+ trades
}

CONFIG_PATH = r"C:\Rebel Technologies\Rebel Master\Config\master_config.yaml"
TRAINING_DATASET_PATH = r"C:\Rebel Technologies\Rebel Master\ML\training_dataset.csv"


class RebelEngine:
    def __init__(self):
        self.config = self.load_config()
        self.ai_config = self.config.get("ai", {}) or {}
        self.rules_only = bool(self.config.get("rules_only", False))
        if self.rules_only:
            # Force-disable AI/GPT when validating core rules
            self.ai_config["enabled"] = False
            self.ai_config["gpt_enabled"] = False
            print("[MODE] Rules-only enabled: AI/GPT disabled")
        self.gpt_enabled = bool(self.ai_config.get("gpt_enabled", False))
        self.gpt_mode = "off"
        self.ai_model_primary = self.ai_config.get("gpt_model_primary", "gpt-4o-mini")
        self.ai_model_secondary = self.ai_config.get("gpt_model_secondary", "gpt-4o")
        self.ai_secondary_split = float(self.ai_config.get("gpt_secondary_split", 0.20))
        self.ai_dual_model_start = int(self.ai_config.get("gpt_dual_model_start", 1500))
        self.ai_max_tokens = int(self.ai_config.get("gpt_max_tokens", 200))
        self.ai_temperature = float(self.ai_config.get("gpt_temperature", 0.1))
        api_key = self.ai_config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if api_key and HAS_OPENAI and openai:
            openai.api_key = api_key
        self.scanner_config = self.config.get("scanner", {}) or {}
        self.min_score = int(self.scanner_config.get("min_score", 5))  # Higher threshold for noise reduction
        self.history_bars = int(self.scanner_config.get("history_bars", 200))
        self.timeframe = self._resolve_timeframe(self.scanner_config.get("timeframe", "H1"))

        # Intelligent Scanner (confidence layer)
        is_cfg = self.config.get("intelligent_scanner", {}) or {}
        self.intelligent_scanner_enabled = bool(is_cfg.get("enabled", False)) and HAS_INTELLIGENT_SCANNER
        self.intelligent_scanner = None
        self.scanner_confidence_thresholds = is_cfg.get("confidence_thresholds", {})
        self.scanner_global_min_conf = int(is_cfg.get("min_confidence", 65))
        if self.intelligent_scanner_enabled:
            try:
                self.intelligent_scanner = RebelIntelligentScanner(self.config)
                self.intelligent_scanner.adapter._connected = True
                self.intelligent_scanner.connected = True
                print(f"[SCANNER] Intelligent Scanner ACTIVE (mode={is_cfg.get('mode', 'ta_only')}, min_conf={self.scanner_global_min_conf}%)")
            except Exception as e:
                print(f"[SCANNER] Failed to initialize Intelligent Scanner: {e}")
                self.intelligent_scanner_enabled = False

        # Core validation progress (closed trades)
        core_cfg = self.config.get("core_validation", {}) or {}
        self.core_validation_enabled = bool(core_cfg.get("enabled", False))
        self.core_validation_target = int(core_cfg.get("target_trades", 500))
        self._last_core_validation_count = None

        # Indicator defaults (can be overridden in scanner section of YAML)
        self.ema_fast = int(self.scanner_config.get("ema_fast", 9))
        self.ema_slow = int(self.scanner_config.get("ema_slow", 21))
        self.rsi_period = int(self.scanner_config.get("rsi_period", 14))
        self.adx_period = int(self.scanner_config.get("adx_period", 14))
        self.atr_period = int(self.scanner_config.get("atr_period", 14))
        
        # Broker connector (set via set_broker)
        self.broker = None
        
        # ML Feature/Label logging (disabled during core validation)
        self.ml_feature_logging_enabled = bool(
            (self.config.get("ml_feature_logging", {}) or {}).get("enabled", False)
        )
        if self.rules_only:
            self.ml_feature_logging_enabled = False
        self.ml_logger = MLFeatureLogger(base_path=r"C:\Rebel Technologies\Rebel Master\ML") if self.ml_feature_logging_enabled else None
        self.feature_logger = self.ml_logger  # Alias for compatibility
        
        # ML Label Generator for trade outcomes (disabled during core validation)
        self.ml_labeler = MLLabelGenerator(base_path=r"C:\Rebel Technologies\Rebel Master\ML") if self.ml_feature_logging_enabled else None
        self.label_generator = self.ml_labeler  # Alias for compatibility
        
        # Position Manager for tracking trades and generating labels
        self.position_manager = RebelPositionManager(ml_labeler=self.ml_labeler)
        
        # Profit lock config (advanced tier-based trailing)
        self.profit_lock_config = self.config.get("profit_lock", {})
        
        # ATR cache for profit lock calculations
        self._atr_cache = {}
        
        # Points-based Profit Lock Manager (+10->BE, +40->+20, etc.)
        self.profit_lock_manager = ProfitLockManager()
        print("[ENGINE] Points-based Profit Lock Ladder ACTIVE")
        
        # ML Trade Filter - Config Driven
        self.ml_filter = None
        self.ml_filter_config = self.config.get("ml_filter", {})
        if self.rules_only:
            # Disable ML filter + shadow mode in rules-only runs
            self.ml_filter_config["enabled"] = False
        self.ml_enabled = self.ml_filter_config.get("enabled", False)
        self.ml_shadow_mode = not self.ml_enabled  # shadow_mode = True means don't block
        if self.rules_only:
            self.ml_shadow_mode = False

        # Filter-skip state (avoid scanning symbols that never pass filters)
        self.filter_skip_config = self.config.get("filter_skip", {}) or {}
        self.filter_skip_enabled = bool(self.filter_skip_config.get("enabled", False))
        self.filter_skip_max_fails = int(self.filter_skip_config.get("max_failed_scans", 30))
        self.filter_skip_state_path = self.filter_skip_config.get(
            "state_file",
            r"C:\Rebel Technologies\Rebel Master\logs\filter_skip_state.json"
        )
        self.filter_skip_state = self._load_filter_skip_state()

        # Auto-purge symbols with poor win rates
        self.symbol_purge_config = self.config.get("symbol_purge", {}) or {}
        self.symbol_purge_enabled = bool(self.symbol_purge_config.get("enabled", False))
        self.symbol_purge_min_trades = int(self.symbol_purge_config.get("min_trades", 10))
        self.symbol_purge_min_trades_by_group = self.symbol_purge_config.get("per_group_min_trades", {}) or {}
        self.symbol_purge_min_losses = int(self.symbol_purge_config.get("min_losses", 6))
        self.symbol_purge_max_win_rate = float(self.symbol_purge_config.get("max_win_rate", 0.35))
        self.symbol_purge_state_path = self.symbol_purge_config.get(
            "state_file",
            r"C:\Rebel Technologies\Rebel Master\logs\trade_limiter_state.json"
        )
        self._last_purge_snapshot = set()

        # Apply trade-count based ML staging (model + thresholds)
        self._apply_ml_staging()
        # Apply GPT staging (observe/decide) based on trade count
        self._apply_gpt_staging()
        
        # Build ML bypass symbol list from config groups
        self.ml_bypass_symbols = set()
        bypass_groups = self.ml_filter_config.get("bypass_groups", [])
        symbols_config = self.config.get("symbols", {})
        groups_config = symbols_config.get("groups", {}) if isinstance(symbols_config, dict) else {}
        
        for group_name in bypass_groups:
            group = groups_config.get(group_name, {})
            if group.get("enabled", True):
                group_symbols = group.get("symbols", [])
                self.ml_bypass_symbols.update(group_symbols)
        
        if self.ml_bypass_symbols:
            print(f"[ML] Bypass enabled for {len(self.ml_bypass_symbols)} symbols from groups: {bypass_groups}")
        
        # Per-group ML thresholds (e.g., FX needs higher win_prob)
        self.ml_group_thresholds = self.ml_filter_config.get("group_thresholds", {})
        if self.ml_group_thresholds:
            print(f"[ML] Per-group thresholds: {self.ml_group_thresholds}")
        
        # Select model based on config
        model_name = self.ml_filter_config.get("model", "rf_v1")
        ml_model_path = ML_MODEL_PATHS.get(model_name, ML_MODEL_PATHS["rf_v1"])
        
        if self.rules_only:
            print("[MODE] Rules-only enabled: ML filter skipped")
        elif ML_AVAILABLE and os.path.exists(ml_model_path):
            try:
                # Build MLFilterConfig from YAML settings
                hard_filters = self.ml_filter_config.get("hard_filters", {})
                ml_config = MLFilterConfig(
                    min_win_prob=float(self.ml_filter_config.get("min_win_prob", 0.50)),
                    max_loss_prob=float(self.ml_filter_config.get("max_loss_prob", 0.70)),
                    max_spread_ratio=float(hard_filters.get("max_spread_ratio", 0.005)),
                    max_spread_ratio_by_class=hard_filters.get("max_spread_ratio_by_class", {}) or {},
                    max_spread_vol_ratio=float(hard_filters.get("max_spread_vol_ratio", 0.25)),
                    min_reward_risk=float(hard_filters.get("min_reward_risk", 1.2)),
                    block_midzone_rsi=bool(hard_filters.get("block_rsi_midzone", False)),
                    return_reasons=True
                )
                self.ml_filter = MLTradeFilter(ml_model_path, ml_config)
                
                if self.ml_enabled:
                    print(f"[ML] {model_name.upper()} loaded - FILTER MODE ACTIVE (blocking bad trades)")
                else:
                    print(f"[ML] {model_name.upper()} loaded - SHADOW MODE (logging only)")
            except Exception as e:
                print(f"[ML] Failed to load model: {e}")
                self.ml_filter = None
        else:
            print(f"[ML] Model not found at {ml_model_path} - ML filter disabled")

    def _load_filter_skip_state(self) -> dict:
        """Load filter-skip state from disk."""
        if not self.filter_skip_enabled:
            return {"symbols": {}}
        try:
            if os.path.exists(self.filter_skip_state_path):
                with open(self.filter_skip_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "symbols" in data:
                    return data
        except Exception as e:
            print(f"[FILTER] Failed to load skip state: {e}")
        return {"symbols": {}}

    def _save_filter_skip_state(self) -> None:
        """Persist filter-skip state."""
        if not self.filter_skip_enabled:
            return
        try:
            os.makedirs(os.path.dirname(self.filter_skip_state_path), exist_ok=True)
            with open(self.filter_skip_state_path, "w", encoding="utf-8") as f:
                json.dump(self.filter_skip_state, f, indent=2)
        except Exception as e:
            print(f"[FILTER] Failed to save skip state: {e}")

    def _get_symbol_purge_blocklist(self) -> set:
        """Return symbols to purge based on win/loss stats."""
        if not self.symbol_purge_enabled:
            return set()
        try:
            if not os.path.exists(self.symbol_purge_state_path):
                return set()
            with open(self.symbol_purge_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            symbol_stats = state.get("symbol_stats", {}) or {}
        except Exception as e:
            print(f"[PURGE] Failed to load limiter state: {e}")
            return set()

        blocked = set()
        details = []
        for symbol, stats in symbol_stats.items():
            wins = int(stats.get("wins", 0))
            losses = int(stats.get("losses", 0))
            total = wins + losses
            group = self._get_asset_group(symbol)
            group_min_trades = int(self.symbol_purge_min_trades_by_group.get(group, self.symbol_purge_min_trades))
            if total < group_min_trades:
                continue
            if losses < self.symbol_purge_min_losses:
                continue
            win_rate = (wins / total) if total > 0 else 0.0
            if win_rate < self.symbol_purge_max_win_rate:
                blocked.add(symbol)
                details.append({
                    "symbol": symbol,
                    "group": group,
                    "wins": wins,
                    "losses": losses,
                    "total": total,
                    "min_trades": group_min_trades,
                    "win_rate": round(win_rate, 4),
                })
        # Persist a readable list when it changes
        if blocked != self._last_purge_snapshot:
            try:
                out_path = r"C:\Rebel Technologies\Rebel Master\logs\purged_symbols.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "count": len(blocked),
                        "symbols": sorted(list(blocked)),
                        "details": details,
                        "generated_at": datetime.now().isoformat(),
                    }, f, indent=2)
                print(f"[PURGE] Active purged symbols: {len(blocked)} (see logs/purged_symbols.json)")
            except Exception as e:
                print(f"[PURGE] Failed to write purged_symbols.json: {e}")
            self._last_purge_snapshot = set(blocked)
        return blocked

    def _record_filter_result(self, symbol: str, passed: bool) -> None:
        """Update filter-skip stats for a symbol."""
        if not self.filter_skip_enabled or not symbol:
            return
        symbols = self.filter_skip_state.setdefault("symbols", {})
        entry = symbols.get(symbol, {"fail_count": 0, "ever_passed": False, "last_pass": None})
        if passed:
            entry["fail_count"] = 0
            entry["ever_passed"] = True
            entry["last_pass"] = datetime.now().isoformat()
        else:
            entry["fail_count"] = int(entry.get("fail_count", 0)) + 1
        symbols[symbol] = entry
        self.filter_skip_state["symbols"] = symbols
        self._save_filter_skip_state()

    def _get_filter_skip_blocklist(self) -> set:
        """Return symbols to skip scanning (never passed filters)."""
        if not self.filter_skip_enabled:
            return set()
        symbols = self.filter_skip_state.get("symbols", {})
        blocked = {
            sym for sym, data in symbols.items()
            if not data.get("ever_passed") and int(data.get("fail_count", 0)) >= self.filter_skip_max_fails
        }
        return blocked

    def _get_training_trade_count(self) -> int:
        """
        Count merged trades in training_dataset.csv (header excluded).
        Returns 0 if file missing or unreadable.
        """
        try:
            if not os.path.exists(TRAINING_DATASET_PATH):
                return 0
            with open(TRAINING_DATASET_PATH, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f) - 1
            return max(count, 0)
        except Exception as e:
            print(f"[ML] Trade count read error: {e}")
            return 0

    def _apply_ml_staging(self) -> None:
        """
        Apply model/threshold staging based on training trade count.
        This overrides ml_filter.model, min_win_prob, and group_thresholds.
        """
        staging = self.ml_filter_config.get("staging", {}) or {}
        if not staging.get("enabled", False):
            return

        trade_count = self._get_training_trade_count()
        stages = staging.get("stages", []) or []
        if not stages:
            return

        selected_stage = None
        for stage in stages:
            min_trades = int(stage.get("min_trades", 0))
            max_trades = stage.get("max_trades")
            if trade_count < min_trades:
                continue
            if max_trades is None or trade_count <= int(max_trades):
                selected_stage = stage
                break

        if not selected_stage:
            selected_stage = stages[-1]

        # Resolve model override
        stage_model = selected_stage.get("model")
        if stage_model:
            self.ml_filter_config["model"] = stage_model

        # Resolve min_win_prob (supports ramp)
        min_win_prob = selected_stage.get("min_win_prob")
        start_prob = selected_stage.get("min_win_prob_start")
        end_prob = selected_stage.get("min_win_prob_end")
        min_trades = int(selected_stage.get("min_trades", 0))
        max_trades = selected_stage.get("max_trades")

        if start_prob is not None and end_prob is not None and max_trades is not None:
            span = max(int(max_trades) - min_trades, 1)
            progress = min(max(trade_count - min_trades, 0), span) / span
            min_win_prob = float(start_prob) + (float(end_prob) - float(start_prob)) * progress

        if min_win_prob is not None:
            self.ml_filter_config["min_win_prob"] = round(float(min_win_prob), 4)

        # Resolve group thresholds (supports ramp or per-group map)
        stage_group_thresholds = selected_stage.get("group_thresholds")
        group_threshold = selected_stage.get("group_threshold")
        group_start = selected_stage.get("group_threshold_start")
        group_end = selected_stage.get("group_threshold_end")
        if group_start is not None and group_end is not None and max_trades is not None:
            span = max(int(max_trades) - min_trades, 1)
            progress = min(max(trade_count - min_trades, 0), span) / span
            group_threshold = float(group_start) + (float(group_end) - float(group_start)) * progress
        if isinstance(stage_group_thresholds, dict) and stage_group_thresholds:
            current_groups = self.ml_filter_config.get("group_thresholds", {}) or {
                "forex": 0.0,
                "crypto": 0.0,
                "indices": 0.0,
                "metals": 0.0,
                "energies": 0.0,
                "softs": 0.0,
            }
            merged = dict(current_groups)
            for k, v in stage_group_thresholds.items():
                merged[k] = round(float(v), 4)
            self.ml_filter_config["group_thresholds"] = merged
        elif group_threshold is not None:
            current_groups = self.ml_filter_config.get("group_thresholds", {})
            if not current_groups:
                current_groups = {
                    "forex": 0.0,
                    "crypto": 0.0,
                    "indices": 0.0,
                    "metals": 0.0,
                    "energies": 0.0,
                    "softs": 0.0,
                }
            self.ml_filter_config["group_thresholds"] = {
                k: round(float(group_threshold), 4) for k in current_groups.keys()
            }

        stage_name = selected_stage.get("name", "unknown")
        print(f"[ML] Staging active: stage={stage_name} trades={trade_count} "
              f"model={self.ml_filter_config.get('model')} "
              f"min_win_prob={self.ml_filter_config.get('min_win_prob')}")

    def _apply_gpt_staging(self) -> None:
        """
        Set GPT mode based on trade count:
        - observe at gpt_observe_at trades
        - decide between gpt_decide_start and gpt_decide_end
        - dual_model at gpt_dual_model_start and above
        """
        if not self.gpt_enabled:
            self.gpt_mode = "off"
            return

        trade_count = self._get_training_trade_count()
        observe_at = int(self.ai_config.get("gpt_observe_at", 1000))
        decide_start = int(self.ai_config.get("gpt_decide_start", 1250))
        decide_end = int(self.ai_config.get("gpt_decide_end", 2000))
        dual_model_start = int(self.ai_config.get("gpt_dual_model_start", 1500))

        if trade_count >= dual_model_start:
            self.gpt_mode = "dual_model"
        elif decide_start <= trade_count <= decide_end:
            self.gpt_mode = "decide"
        elif trade_count >= observe_at:
            self.gpt_mode = "observe"
        else:
            self.gpt_mode = "off"

        self.ai_config["gpt_mode"] = self.gpt_mode
        print(f"[AI] GPT mode={self.gpt_mode} trades={trade_count}")

    def _gpt_trade_decision(self, signal: dict, group_threshold: float) -> dict | None:
        """
        Ask GPT to approve or reject a trade.
        Returns {"decision": "approve"|"reject", "reason": "..."} or None when disabled.
        """
        if not self.gpt_enabled or self.gpt_mode == "off":
            return None
        if not HAS_OPENAI or openai is None or not getattr(openai, "api_key", None):
            return {"decision": "reject", "reason": "gpt_unavailable"}

        symbol = signal.get("symbol", "")
        direction = (signal.get("direction", "") or "").lower()
        score = signal.get("score", 0)
        score_100 = signal.get("score_100")
        if score_100 is None:
            score_100 = int(round((score / 5.0) * 100)) if isinstance(score, (int, float)) else 0
        score_ratio = (score / 5.0) if isinstance(score, (int, float)) else 0.0
        indicators = signal.get("indicators", {}) or {}

        payload = {
            "symbol": symbol,
            "direction": direction,
            "score": score,
            "score_ratio": round(score_ratio, 4),
            "group_threshold": round(float(group_threshold), 4),
            "trend_bias": signal.get("trend_bias"),
            "risk_valid": signal.get("risk_valid"),
            "risk_messages": signal.get("risk_messages"),
            "counter_trend": bool(signal.get("counter_trend", False)),
            "indicators": {
                "ema9": indicators.get("ema9"),
                "ema21": indicators.get("ema21"),
                "ema50": indicators.get("ema50"),
                "rsi": indicators.get("rsi"),
                "atr": indicators.get("atr"),
                "adx": indicators.get("adx"),
            },
        }

        prompt = (
            "You are REBEL MASTER GPT MINI. You must decide if this trade should be taken.\n"
            "Return ONLY valid JSON with exact fields:\n"
            '{ "decision": "approve" | "reject", "reason": "short reason" }\n'
            "Be conservative. Reject if unsure. Data:\n"
            f"{json.dumps(payload, indent=2)}"
        )

        try:
            trade_count = self._get_training_trade_count()
            use_secondary = False
            if self.gpt_mode in ("dual_model", "deploy") and trade_count >= self.ai_dual_model_start:
                # Deterministic split by symbol+direction
                key = f"{symbol}:{direction}"
                bucket = (abs(hash(key)) % 100) / 100.0
                use_secondary = bucket < self.ai_secondary_split
            model = self.ai_model_secondary if use_secondary else self.ai_model_primary

            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a trading risk gate. Respond only with JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.ai_max_tokens,
                temperature=self.ai_temperature,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            decision = data.get("decision", "").lower()
            if decision not in ("approve", "reject"):
                return {"decision": "reject", "reason": "invalid_gpt_decision"}
            result = {"decision": decision, "reason": data.get("reason", "")}

            # Optional: dual-observe at configured threshold (observe mode only)
            observe_dual_at = self.ai_config.get("gpt_observe_dual_at")
            if self.gpt_mode == "observe" and observe_dual_at is not None:
                try:
                    observe_dual_at = int(observe_dual_at)
                except Exception:
                    observe_dual_at = None
            if self.gpt_mode == "observe" and observe_dual_at is not None and trade_count >= observe_dual_at:
                try:
                    secondary_response = openai.chat.completions.create(
                        model=self.ai_model_secondary,
                        messages=[
                            {"role": "system", "content": "You are a trading risk gate. Respond only with JSON."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=self.ai_max_tokens,
                        temperature=self.ai_temperature,
                    )
                    secondary_content = secondary_response.choices[0].message.content.strip()
                    if secondary_content.startswith("```"):
                        secondary_content = secondary_content.split("```")[1]
                        if secondary_content.startswith("json"):
                            secondary_content = secondary_content[4:]
                    secondary_data = json.loads(secondary_content)
                    secondary_decision = secondary_data.get("decision", "").lower()
                    if secondary_decision in ("approve", "reject"):
                        result["secondary"] = {
                            "decision": secondary_decision,
                            "reason": secondary_data.get("reason", "")
                        }
                except Exception:
                    pass
            return result
        except Exception as e:
            return {"decision": "reject", "reason": f"gpt_error:{e}"}

    def set_broker(self, broker):
        """
        Set the broker connector for the engine.
        
        Args:
            broker: BrokerConnector instance
        """
        self.broker = broker

    # -------------------------------------------------------------------------
    # ASSET CLASSIFICATION & ML THRESHOLDS
    # -------------------------------------------------------------------------
    def _get_asset_group(self, symbol: str) -> str:
        """Classify symbol into asset group for ML threshold lookup."""
        s = symbol.upper()
        
        # Crypto
        crypto_tokens = ("BTC", "ETH", "XRP", "LTC", "ADA", "DOG", "DOT", "XLM", "SOL",
                         "AVAX", "AAVE", "BNB", "SAND", "UNI", "XTZ", "BCH", "COMP",
                         "CRV", "KSM", "LNK", "LRC", "MANA", "SUSHI", "BAT")
        if any(p in s for p in crypto_tokens):
            return "crypto"
        
        # Indices
        if any(k in s for k in ("US500", "US30", "US2000", "USTECH", "NAS100", "DAX40",
                                "SPA35", "UK100", "HK50", "CHINA50", "AUS200", "EU50",
                                "FRA40", "JPN225", "NETH25", "SWI20", "VIX", "USDINDEX",
                                "GER40", "IT40", "SGFREE", "CAC40", "EUSTX50", "HSI",
                                "NK225", "DJ30", "FT100", "SPI200", "S&P", "CN50")):
            return "indices"
        
        # Energies
        if any(k in s for k in ("UKOIL", "USOIL", "BRENT", "WTI", "NATGAS", "OIL", "GAS")):
            return "energies"
        
        # Metals
        if any(k in s for k in ("XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER", "COPPER")):
            return "metals"
        
        # Softs
        if any(k in s for k in ("COCOA", "COFFEE", "SOYBEAN", "SUGAR", "COTTON", "WHEAT", "CORN")):
            return "softs"
        
        # Default to forex
        return "forex"
    
    def _get_scanner_confidence_min(self, symbol: str) -> int:
        """Get the minimum confidence threshold for a symbol's family."""
        group = self._get_asset_group(symbol)
        # Map generic groups to config keys (fx -> fx_major/fx_minor/fx_exotic)
        s = symbol.upper()
        if group == "forex":
            exotic_tokens = ("ZAR", "MXN", "PLN", "CZK", "HUF", "SEK", "NOK", "SGD",
                             "THB", "RON", "BRL", "CLP", "COP", "IDR", "KRW", "TWD",
                             "INR", "HKD", "CNH")
            if any(t in s for t in exotic_tokens):
                return int(self.scanner_confidence_thresholds.get("fx_exotic", self.scanner_global_min_conf))
            major_pairs = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD")
            if s in major_pairs:
                return int(self.scanner_confidence_thresholds.get("fx_major", self.scanner_global_min_conf))
            return int(self.scanner_confidence_thresholds.get("fx_minor", self.scanner_global_min_conf))
        return int(self.scanner_confidence_thresholds.get(group, self.scanner_global_min_conf))

    def _get_ml_threshold(self, symbol: str) -> float:
        """Get the ML win_prob threshold for a symbol based on its asset group."""
        group = self._get_asset_group(symbol)
        default_threshold = float(self.ml_filter_config.get("min_win_prob", 0.50))
        return float(self.ml_group_thresholds.get(group, default_threshold))

    def _baseline_locked_pass(self, signal: Dict[str, Any]) -> bool:
        """Apply locked baseline gate to live signals."""
        cfg = self.config.get("baseline_locked", {}) or {}
        if not cfg.get("enabled", False):
            return True

        allow_missing = bool(cfg.get("allow_missing_features", False))
        features = signal.get("features") or {}
        symbol = signal.get("symbol", "UNKNOWN")
        direction = str(signal.get("direction", "")).lower()

        spread_ratio = features.get("spread_ratio")
        if spread_ratio is None:
            if allow_missing:
                return True
            print(f"[BASELINE] {symbol}: BLOCKED - missing spread_ratio")
            return False
        if float(spread_ratio) >= float(cfg.get("spread_max", 0.04)):
            print(f"[BASELINE] {symbol}: BLOCKED - spread_ratio {spread_ratio:.4f} >= {cfg.get('spread_max', 0.04)}")
            return False

        rsi = features.get("rsi")
        if rsi is None:
            if allow_missing:
                return True
            print(f"[BASELINE] {symbol}: BLOCKED - missing rsi")
            return False
        rsi_min = float(cfg.get("rsi_min", 20))
        rsi_max = float(cfg.get("rsi_max", 80))
        if not (rsi_min < float(rsi) < rsi_max):
            print(f"[BASELINE] {symbol}: BLOCKED - rsi {rsi:.2f} outside {rsi_min}-{rsi_max}")
            return False

        rr_min = float(cfg.get("rr_min", 1.2))
        trading_cfg = self.config.get("trading", {}) or {}
        sl_mult = float(trading_cfg.get("sl_atr_multiplier", 0) or 0)
        tp_mult = float(trading_cfg.get("tp_atr_multiplier", 0) or 0)
        rr_ratio = (tp_mult / sl_mult) if sl_mult > 0 else 0.0
        if rr_ratio < rr_min:
            print(f"[BASELINE] {symbol}: BLOCKED - RR {rr_ratio:.2f} < {rr_min}")
            return False

        if bool(cfg.get("ema_bias", True)):
            ema_fast = features.get("ema_fast")
            ema_slow = features.get("ema_slow")
            if ema_fast is None or ema_slow is None:
                if allow_missing:
                    return True
                print(f"[BASELINE] {symbol}: BLOCKED - missing ema_fast/ema_slow")
                return False
            if direction in ("long", "buy") and not (ema_fast > ema_slow):
                print(f"[BASELINE] {symbol}: BLOCKED - ema_fast <= ema_slow for long")
                return False
            if direction in ("short", "sell") and not (ema_fast < ema_slow):
                print(f"[BASELINE] {symbol}: BLOCKED - ema_fast >= ema_slow for short")
                return False

        return True

    # -------------------------------------------------------------------------
    # EXPOSURE CHECK (30% rule)
    # -------------------------------------------------------------------------
    def _exposure_ok(self) -> bool:
        """
        Check if margin exposure is under 30% limit.
        Returns False if we should block new trades.
        """
        account = self.broker.get_account_info()
        if not account:
            print("[RISK] Cannot get account info. Blocking trades.")
            return False
        
        equity = account.get("equity", 0)
        free_margin = account.get("free_margin", 0)
        
        if equity <= 0:
            print("[RISK] No equity. Blocking trades.")
            return False
        
        margin_used = equity - free_margin  # how much is "in use"
        margin_limit = equity * 0.30        # 30% exposure allowed
        
        if margin_used >= margin_limit:
            print(
                f"[RISK] Exposure limit reached. "
                f"Used=${margin_used:.2f}, Limit=${margin_limit:.2f}. "
                f"Blocking new trades."
            )
            return False
        
        return True

    # -------------------------------------------------------------------------
    # PROFIT LOCK SYSTEM - Advanced ATR-based tiered trailing
    # -------------------------------------------------------------------------
    def _get_atr_for_symbol(self, symbol: str, timeframe_str: str = "H1") -> float:
        """
        Get ATR for a symbol, with caching to avoid repeated calculations.
        """
        cache_key = f"{symbol}_{timeframe_str}"
        
        # Use cached value if fresh (within last 60 seconds)
        if cache_key in self._atr_cache:
            cached_val, cached_time = self._atr_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < 60:
                return cached_val
        
        # Calculate fresh ATR
        tf = self._resolve_timeframe(timeframe_str)
        
        if not mt5.symbol_select(symbol, True):
            return 0.0
        
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, 50)
        if rates is None or len(rates) < 20:
            return 0.0
        
        df = pd.DataFrame(rates)
        atr_series = self._compute_atr(df, 14)
        atr_val = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0
        
        # Cache the result
        self._atr_cache[cache_key] = (atr_val, datetime.now())
        
        return atr_val

    def _compute_profit_lock_sl(self, pos, atr: float, tiers: list, initial_sl_dist_atr: float = 2.0):
        """
        Advanced profit lock: computes new SL based on ATR tiers.
        
        - Uses ATR-based tiers
        - Locks more profit as trade moves further
        - Never loosens SL or crosses current price
        
        Returns:
            New SL price, or None if no adjustment needed
        """
        symbol = pos.get("symbol", "")
        ticket = pos.get("ticket", 0)
        
        # Determine direction: 0=BUY, 1=SELL in MT5
        pos_type = pos.get("type", 0)
        direction = 1 if pos_type == 0 else -1  # 1 for long, -1 for short
        
        entry = pos.get("price_open", 0)
        current_sl = pos.get("sl", 0)
        current_price = pos.get("price_current", 0)
        
        if atr is None or atr <= 0:
            return None
        
        # Move in our favour (in price units)
        move = (current_price - entry) * direction
        if move <= 0:
            return None  # not in profit yet
        
        # Convert move to ATR units
        move_atr = move / atr
        
        # Work out initial 'R' (risk per unit based on initial SL distance)
        initial_sl_price = entry - direction * initial_sl_dist_atr * atr
        initial_r = (entry - initial_sl_price) * direction  # positive if valid
        
        if initial_r <= 0:
            return None
        
        # Pick the highest tier that is triggered
        active_tier = None
        for tier in tiers:
            if move_atr >= tier.get("trigger_atr", 0.0):
                active_tier = tier
            # Don't break - we want the highest matching tier
        
        if not active_tier:
            return None
        
        lock_dist_atr = active_tier.get("lock_distance_atr", 1.0)
        min_lock_rr = active_tier.get("min_lock_rr", 0.0)
        
        # Base SL at ATR distance behind current price
        candidate_sl = current_price - direction * lock_dist_atr * atr
        
        # Enforce min_lock_rr if possible (don't trail too loose after big move)
        target_profit = min_lock_rr * initial_r
        desired_sl = entry + direction * target_profit
        
        if direction > 0:  # long
            candidate_sl = max(candidate_sl, desired_sl)
        else:  # short
            candidate_sl = min(candidate_sl, desired_sl)
        
        # Never cross current price
        if direction > 0 and candidate_sl >= current_price:
            candidate_sl = current_price - 0.1 * atr
        elif direction < 0 and candidate_sl <= current_price:
            candidate_sl = current_price + 0.1 * atr
        
        # Never LOOSEN SL
        if current_sl and current_sl > 0:
            if direction > 0 and candidate_sl <= current_sl:
                return None  # would loosen SL for long
            if direction < 0 and candidate_sl >= current_sl:
                return None  # would loosen SL for short
        
        return candidate_sl

    def _update_profit_locks(self):
        """
        Main profit lock update loop.
        Checks all open positions and tightens SL based on ATR tiers.
        """
        cfg = self.profit_lock_config
        if not cfg or not cfg.get("enabled", False):
            return
        
        min_atr = cfg.get("min_atr", 0.0)
        tiers = cfg.get("tiers", [])
        initial_sl_atr = cfg.get("initial_sl_atr", 2.0)
        timeframe = cfg.get("timeframe", "H1")
        
        if not tiers:
            return
        
        # Ensure tiers are sorted by trigger_atr (ascending)
        tiers = sorted(tiers, key=lambda t: t.get("trigger_atr", 0.0))
        
        # Get open positions
        if not self.broker:
            return
        
        positions = self.broker.get_positions()
        if not positions:
            return
        
        for pos in positions:
            symbol = pos.get("symbol", "")
            ticket = pos.get("ticket", 0)
            
            # Get ATR for this symbol
            atr = self._get_atr_for_symbol(symbol, timeframe)
            if atr is None or atr < min_atr:
                continue
            
            # Compute new SL
            new_sl = self._compute_profit_lock_sl(pos, atr, tiers, initial_sl_atr)
            if new_sl is None:
                continue
            
            # Get symbol digits for rounding
            symbol_info = mt5.symbol_info(symbol)
            digits = symbol_info.digits if symbol_info else 5
            new_sl = round(new_sl, digits)
            
            # Modify position SL
            try:
                # Use MT5 directly if broker doesn't have modify method
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": ticket,
                    "sl": new_sl,
                    "tp": pos.get("tp", 0),
                }
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    direction_str = "LONG" if pos.get("type", 0) == 0 else "SHORT"
                    print(f"[LOCK] {symbol} #{ticket} ({direction_str}): SL tightened to {new_sl:.5f} (ATR={atr:.5f})")
                else:
                    error = result.comment if result else "Unknown error"
                    print(f"[LOCK] Failed to update SL for {symbol} #{ticket}: {error}")
            except Exception as e:
                print(f"[LOCK] Error updating SL for {symbol} #{ticket}: {e}")

    # -------------------------------------------------------------------------
    # Core setup
    # -------------------------------------------------------------------------
    def load_config(self):
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    def _resolve_timeframe(self, tf_value):
        """
        Map string timeframe (from YAML) to MT5 constant.
        Accepts values like: M1, M5, M15, M30, H1, H4, D1, etc.
        Defaults to H1 if unknown.
        """
        if isinstance(tf_value, int):
            # Already an MT5 constant
            return tf_value

        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return tf_map.get(str(tf_value).upper(), mt5.TIMEFRAME_H1)

    def connect_mt5(self):
        cfg = self.config.get("mt5", {})
        path = cfg.get("path")
        
        print(f"[MT5] Connecting to terminal: {path}")
        if not mt5.initialize(path=path):
            raise RuntimeError(f"MT5 failed to initialize: {mt5.last_error()}")

        # Check if already logged into correct account
        account_info = mt5.account_info()
        login = cfg.get("login")
        
        if account_info and account_info.login == login:
            print(f"[MT5] Already logged into account {login} - skipping login")
        elif login:
            # Only try login if not already on correct account
            password = cfg.get("password")
            server = cfg.get("server")
            if password and server:
                print(f"[MT5] Logging into account {login}...")
                authorized = mt5.login(login=login, password=password, server=server)
                if not authorized:
                    raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

        # Final verification
        account_info = mt5.account_info()
        if account_info:
            print(f"[MT5] Connected: Account {account_info.login} | Balance: {account_info.balance}")
        else:
            print("[MT5] Connected (no account info available)")
        
        return True

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------
    def start(self):
        print("REBEL ENGINE - CLEAN START")
        self.connect_mt5()
        symbols = self._get_symbols_from_config()

        print(f"Scanner ready. Timeframe={self.timeframe}, History bars={self.history_bars}, "
              f"Min score={self.min_score}, Symbols={len(symbols)}")

        self.scan_symbols(symbols)

        # Optional shutdown if this is a one-shot run:
        mt5.shutdown()
        print("Engine finished. MT5 shutdown.")

    def run(self, scanner, executor, notifier, interval: int = 60, 
            min_score: int = 3, strategy_mode: str = "normal", training_mode: bool = False):
        """
        Main run loop with scanner, executor, and notifier integration.
        
        Args:
            scanner: RebelScanner instance for signal generation
            executor: RebelTradeExecutor instance for trade execution
            notifier: RebelNotifier instance for alerts
            interval: Scan interval in seconds
            min_score: Minimum score threshold for signals
            strategy_mode: Strategy mode (conservative/normal/aggressive)
            training_mode: If True, logs ML features for dataset building
        """
        import time
        
        self.training_mode = training_mode
        
        print(f"[ENGINE] Starting main loop (interval={interval}s, mode={strategy_mode})")
        if training_mode:
            print(f"[ENGINE] Training mode ENABLED - logging ML features to {self.ml_logger.base_path}")
        
        # Connect to MT5
        self.connect_mt5()
        
        # Sync position manager with MT5 (recover any positions opened/closed while bot was down)
        try:
            self.position_manager.startup_sync()
        except Exception as e:
            print(f"[ENGINE] Position manager startup sync error: {e}")
        
        # Load risk engine settings
        risk_engine_cfg = self.config.get("risk_engine", {})
        risk_per_trade = risk_engine_cfg.get("percent_risk_per_trade", 3.0)
        max_equity_exposure_pct = risk_engine_cfg.get("max_equity_exposure_pct", 30.0) / 100.0
        
        print(f"[ENGINE] === RISK RULES ===")
        print(f"[ENGINE] Rule 1: {risk_per_trade}% risk per trade (via tick_value/tick_size)")
        print(f"[ENGINE] Rule 2: {max_equity_exposure_pct*100:.0f}% max equity exposure")
        print(f"[ENGINE] Rule 3: 1 position per symbol max")
        print(f"[ENGINE] ===================")
        print(f"[ENGINE] Running in {strategy_mode} mode (min_score={min_score})")
        
        running = True
        scan_count = 0
        position_update_interval = 10  # Update positions every 10 seconds
        last_position_update = 0
        
        while running:
            try:
                scan_count += 1
                print(f"\n[ENGINE] Scan #{scan_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # ---- POSITION MANAGEMENT (sync & detect closures) ----
                try:
                    if mt5.terminal_info() is None:
                        print("[MT5] Terminal not ready - reconnecting...")
                        try:
                            self.connect_mt5()
                        except Exception as e:
                            print(f"[MT5] Reconnect failed: {e}")
                    pos_summary = self.position_manager.manage_positions()
                    if pos_summary["closed_count"] > 0:
                        print(f"[ENGINE] Total trades closed: {pos_summary['closed_count']} | "
                              f"Lifetime P/L: {pos_summary['lifetime_profit_points']:.5f} pts")
                    if self.core_validation_enabled:
                        core_count = int(pos_summary.get("closed_count", 0) or 0)
                        if self._last_core_validation_count != core_count:
                            print(f"[CORE] Validating core: {core_count} / {self.core_validation_target}")
                            self._last_core_validation_count = core_count
                except Exception as e:
                    print(f"[ENGINE] Position management error: {e}")
                
                # ---- PROFIT LOCK SYSTEM (points-based ladder: +10->BE, +40->+20, etc.) ----
                try:
                    positions = self.broker.get_positions()
                    if positions:
                        # Convert to dict format if needed
                        pos_dicts = []
                        for p in positions:
                            if isinstance(p, dict):
                                pos_dicts.append(p)
                            else:
                                # Handle MT5 position tuple/namedtuple
                                pos_dicts.append({
                                    "ticket": p.ticket if hasattr(p, 'ticket') else p.get('ticket', 0),
                                    "symbol": p.symbol if hasattr(p, 'symbol') else p.get('symbol', ''),
                                    "type": p.type if hasattr(p, 'type') else p.get('type', 0),
                                    "price_open": p.price_open if hasattr(p, 'price_open') else p.get('price_open', 0),
                                    "price_current": p.price_current if hasattr(p, 'price_current') else p.get('price_current', 0),
                                    "sl": p.sl if hasattr(p, 'sl') else p.get('sl', 0),
                                    "tp": p.tp if hasattr(p, 'tp') else p.get('tp', 0),
                                    "profit": p.profit if hasattr(p, 'profit') else p.get('profit', 0.0),
                                    "volume": p.volume if hasattr(p, 'volume') else p.get('volume', 0.0),
                                })
                        
                        # Evaluate profit locks
                        actions = self.profit_lock_manager.update_all_positions(pos_dicts)
                        
                        # Execute any lock actions (modify SL or close)
                        if actions:
                            results = self.profit_lock_manager.execute_lock_actions(actions)
                            for r in results:
                                if r.get("success"):
                                    print(f"[LOCK] {r['symbol']} #{r['ticket']}: {r['action']} successful")
                except Exception as e:
                    print(f"[ENGINE] Profit lock update error: {e}")
                
                # ---- RUN SCANNER ----
                if hasattr(scanner, "set_symbol_blocklist"):
                    blocklist = set()
                    if self.filter_skip_enabled:
                        blocklist |= self._get_filter_skip_blocklist()
                    if self.symbol_purge_enabled:
                        blocklist |= self._get_symbol_purge_blocklist()
                    scanner.set_symbol_blocklist(blocklist)
                signals = scanner.scan()
                
                if signals:
                    print(f"[ENGINE] Found {len(signals)} signal(s)")
                    
                    for signal in signals:
                        symbol = signal.get("symbol", "")
                        direction = signal.get("direction", "")
                        score = signal.get("score", 0)
                        score_100 = signal.get("score_100")
                        if score_100 is None:
                            score_100 = int(round((score / 5.0) * 100)) if isinstance(score, (int, float)) else 0
                            signal["score_100"] = score_100
                        
                        # Execute the trade if auto-trade is enabled
                        if executor.auto_trade:
                            # ============================================
                            # RULE 2: 30% EXPOSURE CHECK
                            # ============================================
                            if not self._exposure_ok():
                                print("[ENGINE] Exposure guard triggered. Skipping all new trades.")
                                continue
                            
                            # ============================================
                            # RULE 3: 1 TRADE PER SYMBOL
                            # ============================================
                            open_positions = self.broker.get_positions()
                            open_symbols = {pos["symbol"] for pos in open_positions}
                            
                            if symbol in open_symbols:
                                print(f"[RISK] Skipping {symbol}: already 1 open trade.")
                                continue
                            
                            # ============================================
                            # 5-GATE SIGNAL FILTERS (can be disabled via config)
                            # ============================================
                            signal_filters_enabled = self.config.get("signal_filters", {}).get("enabled", True)
                            if signal_filters_enabled:
                                try:
                                    # Fetch data for filters
                                    htf_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 50)
                                    entry_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
                                    
                                    if htf_data is not None and entry_data is not None:
                                        htf_df = pd.DataFrame(htf_data)
                                        entry_df = pd.DataFrame(entry_data)
                                        
                                        # Get current spread
                                        tick = mt5.symbol_info_tick(symbol)
                                        current_spread = (tick.ask - tick.bid) if tick else 0.0
                                        
                                        # Run all 5 filters
                                        filters_passed, rejections, soft_flags = run_all_filters(
                                            symbol=symbol,
                                            direction=direction,
                                            htf_data=htf_df,
                                            entry_candles=entry_df,
                                            current_spread=current_spread,
                                            config=self.config
                                        )
                                        
                                        if not filters_passed:
                                            print(f"[FILTER] {symbol}: BLOCKED - {rejections}")
                                            self._record_filter_result(symbol, passed=False)
                                            continue
                                        if "counter_trend" in soft_flags:
                                            signal["counter_trend"] = True
                                            signal["position_scale"] = 0.5
                                    else:
                                        print(f"[FILTER] {symbol}: Skipped (no data)")
                                        self._record_filter_result(symbol, passed=False)
                                        continue
                                except Exception as e:
                                    print(f"[FILTER] {symbol}: Error - {e}")
                                    self._record_filter_result(symbol, passed=False)
                                    continue
                                self._record_filter_result(symbol, passed=True)
                            else:
                                print(f"[FILTER] {symbol}: 5-gate filters DISABLED — signal passes by rules score only")
                                self._record_filter_result(symbol, passed=True)
                            limiter_record_filter_passed(symbol)

                            # ============================================
                            # SCORE GATE (match score to confidence threshold)
                            # ============================================
                            # Locked baseline gate (spread/RSI/EMA/RR)
                            if not self._baseline_locked_pass(signal):
                                continue
                            if isinstance(score, (int, float)) and score < self.min_score:
                                print(f"[SCORE:BLOCKED] {symbol}: score {score}/5 < min_score {self.min_score}/5")
                                continue
                            group_threshold = self._get_ml_threshold(symbol)
                            score_ratio = (score / 5.0) if isinstance(score, (int, float)) else 0.0
                            if score_ratio < group_threshold:
                                print(f"[SCORE:BLOCKED] {symbol}: score {score}/5 ({score_ratio:.2f}) < {group_threshold:.2f}")
                                continue

                            # ============================================
                            # INTELLIGENT SCANNER CONFIDENCE GATE
                            # ============================================
                            scanner_confidence = None
                            if self.intelligent_scanner_enabled and self.intelligent_scanner:
                                try:
                                    scan_result = self.intelligent_scanner.scan_symbol(symbol)
                                    if scan_result and scan_result.get("final_confidence") is not None:
                                        scanner_confidence = int(scan_result["final_confidence"])
                                        scanner_dir = scan_result.get("direction")
                                        min_conf = self._get_scanner_confidence_min(symbol)
                                        family = self._get_asset_group(symbol)

                                        # Direction disagreement — scanner says opposite or no direction
                                        dir_match = True
                                        if scanner_dir is not None:
                                            engine_dir = direction.lower() if direction else ""
                                            if scanner_dir != engine_dir:
                                                dir_match = False

                                        if scanner_confidence < min_conf:
                                            print(f"[CONFIDENCE:BLOCKED] {symbol} ({family}): scanner confidence {scanner_confidence}% < {min_conf}%")
                                            continue
                                        elif not dir_match and scanner_dir is not None:
                                            print(f"[CONFIDENCE:WARN] {symbol}: direction mismatch (engine={direction}, scanner={scanner_dir}) conf={scanner_confidence}%")
                                            # Allow trade but flag it
                                            signal["direction_mismatch"] = True
                                            signal["position_scale"] = signal.get("position_scale", 1.0) * 0.5

                                        print(f"[CONFIDENCE:OK] {symbol} ({family}): confidence={scanner_confidence}% (min={min_conf}%)")
                                        signal["scanner_confidence"] = scanner_confidence
                                    else:
                                        print(f"[CONFIDENCE] {symbol}: scanner returned no result — passing on score only")
                                except Exception as e:
                                    print(f"[CONFIDENCE] {symbol}: scanner error — {e} — passing on score only")

                            # ============================================
                            # GPT DECISION GATE (observe / decide / deploy)
                            # ============================================
                            gpt_decision = self._gpt_trade_decision(signal, group_threshold)
                            if gpt_decision is not None:
                                decision = gpt_decision.get("decision", "")
                                reason = gpt_decision.get("reason", "")
                                print(f"[GPT:{self.gpt_mode.upper()}] {symbol}: {decision} {reason}")
                                secondary = gpt_decision.get("secondary")
                                if secondary:
                                    print(f"[GPT:OBSERVE-SECONDARY] {symbol}: {secondary.get('decision')} {secondary.get('reason')}")
                                if self.gpt_mode in ("decide", "deploy", "dual_model") and decision != "approve":
                                    continue

                            score_100 = signal.get("score_100")
                            if score_100 is None:
                                score_100 = int(round((score / 5.0) * 100)) if isinstance(score, (int, float)) else 0
                            conf_str = f" | conf={scanner_confidence}%" if scanner_confidence is not None else ""
                            print(f"[ENGINE] Signal: {symbol} {direction.upper()} (score={score}/5 | {score_100}/100{conf_str})")

                            # ==========================================================
                            # ML FEATURE LOGGING (only after filters + score gate)
                            # ==========================================================
                            if self.ml_feature_logging_enabled and self.feature_logger and "features" in signal and signal["features"] is not None:
                                try:
                                    self.feature_logger.log_features(
                                        symbol=signal["symbol"],
                                        feature_dict=signal["features"]
                                    )
                                    if self.training_mode:
                                        print(f"[ML_LOGGER] Logged features for {signal['symbol']}")
                                except Exception as e:
                                    print(f"[ML_FEATURE_LOGGER] Error logging features for {signal['symbol']}: {e}")

                            # Notify about the signal
                            if notifier.enabled:
                                notifier.notify_signal(signal)
                            
                            # ============================================
                            # ML FILTER EVALUATION (with bypass for winning asset classes)
                            # Skipped entirely in rules-only mode
                            # ============================================
                            ml_result = None
                            
                            if self.rules_only:
                                pass  # ML filter completely off during core validation
                            elif symbol in self.ml_bypass_symbols:
                                print(f"[ML:BYPASS] {symbol} - bypassing ML filter (winning asset class)")
                            elif self.ml_filter is not None and "features" in signal:
                                try:
                                    # Build candidate from signal features
                                    candidate = signal.get("features", {}).copy()
                                    candidate["symbol"] = symbol
                                    candidate["direction"] = direction
                                    
                                    # Add missing fields required by ML feature engineering
                                    # Get current price for entry_price estimate
                                    tick = mt5.symbol_info_tick(symbol)
                                    if tick:
                                        current_price = tick.ask if direction.lower() == "long" else tick.bid
                                        candidate["entry_price"] = current_price
                                        
                                        # Estimate SL/TP from signal or use ATR-based defaults
                                        atr = signal.get("atr") or candidate.get("atr") or 0.001
                                        if direction.lower() == "long":
                                            candidate["sl"] = current_price - (atr * 2)
                                            candidate["tp"] = current_price + (atr * 3)
                                        else:
                                            candidate["sl"] = current_price + (atr * 2)
                                            candidate["tp"] = current_price - (atr * 3)
                                        
                                        # Default volume (actual volume calculated in executor)
                                        candidate["volume"] = 0.01
                                    
                                    # Encode categorical fields to numeric for ML model
                                    # direction: long=1, short=-1
                                    if isinstance(candidate.get("direction"), str):
                                        candidate["direction"] = 1 if candidate["direction"].lower() == "long" else -1
                                    
                                    # trend_bias: bullish=1, bearish=-1, neutral/unknown=0
                                    tb = candidate.get("trend_bias", "")
                                    if isinstance(tb, str):
                                        if "bull" in tb.lower():
                                            candidate["trend_bias"] = 1
                                        elif "bear" in tb.lower():
                                            candidate["trend_bias"] = -1
                                        else:
                                            candidate["trend_bias"] = 0
                                    
                                    # volatility_regime: expanding=1, contracting=-1, normal=0
                                    vr = candidate.get("volatility_regime", "")
                                    if isinstance(vr, str):
                                        if "expand" in vr.lower():
                                            candidate["volatility_regime"] = 1
                                        elif "contract" in vr.lower():
                                            candidate["volatility_regime"] = -1
                                        else:
                                            candidate["volatility_regime"] = 0
                                    
                                    # Run ML evaluation
                                    if self.ml_shadow_mode:
                                        # Shadow mode: log only, don't block
                                        ml_result = shadow_mode_decision(self.ml_filter, candidate)
                                    else:
                                        # Filter mode: actually evaluate for blocking
                                        ml_result = self.ml_filter.evaluate_candidate(candidate)
                                    
                                    # Log the ML decision
                                    mode_tag = "SHADOW" if self.ml_shadow_mode else "FILTER"
                                    print(f"[ML:{mode_tag}] {symbol}: {ml_result['decision']} | "
                                          f"WinProb={ml_result['win_prob']:.2f} | "
                                          f"LossProb={ml_result['loss_prob']:.2f}")
                                    if ml_result.get('reasons'):
                                        print(f"[ML:{mode_tag}] Reasons: {ml_result['reasons']}")
                                except Exception as e:
                                    print(f"[ML] Evaluation error: {e}")
                            
                            # ============================================
                            # ML FILTER GATE - Block if REJECT or below group threshold
                            # ============================================
                            if ml_result is not None and not self.ml_shadow_mode:
                                # Check per-group threshold (e.g., FX needs 55% instead of 50%)
                                group_threshold = self._get_ml_threshold(symbol)
                                win_prob = ml_result.get('win_prob', 0)
                                
                                if ml_result.get('decision') == 'REJECT':
                                    print(f"[ML:BLOCKED] {symbol} rejected by ML filter - skipping trade")
                                    try:
                                        log_ml_decision(symbol, ml_result, "BLOCKED_BY_ML")
                                    except Exception as e:
                                        print(f"[ML] Log error: {e}")
                                    continue  # Skip to next signal
                                
                                # Per-group threshold check (stricter for FX)
                                elif win_prob < group_threshold:
                                    asset_group = self._get_asset_group(symbol)
                                    print(f"[ML:BLOCKED] {symbol} ({asset_group}) win_prob {win_prob:.2f} < {group_threshold:.2f} threshold")
                                    try:
                                        ml_result['reasons'] = ml_result.get('reasons', []) + [f"group_threshold_{asset_group}"]
                                        log_ml_decision(symbol, ml_result, f"BLOCKED_BY_GROUP_THRESHOLD_{asset_group.upper()}")
                                    except Exception as e:
                                        print(f"[ML] Log error: {e}")
                                    continue  # Skip to next signal
                            
                            # ============================================
                            # ASSET CLASS DAILY LIMIT CHECK
                            # ============================================
                            can_open, limit_reason = limiter_can_trade(symbol)
                            if not can_open:
                                asset_group = self._get_asset_group(symbol)
                                print(f"[LIMITER] {symbol} ({asset_group}): BLOCKED - {limit_reason}")
                                continue
                            
                            # ============================================
                            # RULE 1: 3% RISK PER TRADE (in executor)
                            # ============================================
                            result = executor.open_trade(signal)
                            
                            # Log ML decision with trade outcome (for ACCEPT decisions that proceeded)
                            if ml_result is not None and self.ml_filter_config.get("log_decisions", True):
                                engine_action = "EXECUTED" if result.get("ok") else f"FAILED:{result.get('error')}"
                                try:
                                    log_ml_decision(symbol, ml_result, engine_action)
                                except Exception as e:
                                    print(f"[ML] Log error: {e}")
                            
                            if result.get("ok"):
                                # Record trade for limiter (per-class daily tracking)
                                limiter_record_trade(symbol)
                                print(f"[ENGINE] Trade opened: {symbol} {direction} ticket={result.get('ticket')}")
                                if notifier.enabled:
                                    notifier.notify_trade(result)
                            else:
                                print(f"[ENGINE] Trade failed: {result.get('error')}")
                else:
                    print(f"[ENGINE] No signals this scan")
                
                # Observer-only regime logging (no execution impact)
                try:
                    observe_metals_impulse(self.config)
                except Exception as e:
                    print(f"[REGIME] Observer error: {e}")
                
                # ---- WAIT WITH POSITION UPDATES ----
                print(f"[ENGINE] Next scan in {interval}s...")
                
                # During wait, periodically update position metrics & profit locks
                elapsed = 0
                while elapsed < interval and running:
                    time.sleep(min(position_update_interval, interval - elapsed))
                    elapsed += position_update_interval
                    
                    # Update position metrics (heat, favorable) without full sync
                    try:
                        self.position_manager.update_all_positions()
                    except Exception as e:
                        pass  # Silent fail on position updates
                    
                    # Update profit locks (points-based ladder)
                    try:
                        positions = self.broker.get_positions()
                        if positions:
                            pos_dicts = []
                            for p in positions:
                                if isinstance(p, dict):
                                    pos_dicts.append(p)
                                else:
                                    pos_dicts.append({
                                        "ticket": p.ticket if hasattr(p, 'ticket') else 0,
                                        "symbol": p.symbol if hasattr(p, 'symbol') else '',
                                        "type": p.type if hasattr(p, 'type') else 0,
                                        "price_open": p.price_open if hasattr(p, 'price_open') else 0,
                                        "price_current": p.price_current if hasattr(p, 'price_current') else 0,
                                        "sl": p.sl if hasattr(p, 'sl') else 0,
                                        "tp": p.tp if hasattr(p, 'tp') else 0,
                                        "profit": p.profit if hasattr(p, 'profit') else 0.0,
                                        "volume": p.volume if hasattr(p, 'volume') else 0.0,
                                    })
                            actions = self.profit_lock_manager.update_all_positions(pos_dicts)
                            if actions:
                                self.profit_lock_manager.execute_lock_actions(actions)
                    except Exception as e:
                        pass  # Silent fail on profit lock updates
                
            except KeyboardInterrupt:
                print("\n[ENGINE] Shutdown requested by user")
                running = False
            except Exception as e:
                print(f"[ENGINE] Error during scan: {e}")
                if notifier.enabled:
                    notifier.notify_error(str(e))
                # Continue running after error
                time.sleep(interval)
        
        # ---- FINAL POSITION SYNC (capture any last closures) ----
        try:
            print("[ENGINE] Final position sync...")
            self.position_manager.manage_positions()
            summary = self.position_manager.get_position_summary()
            print(f"[ENGINE] Session Summary: Closed={summary['closed_count']} | "
                  f"Open={summary['open_count']} | Total P/L={summary['lifetime_profit_points']:.5f} pts")
        except Exception as e:
            print(f"[ENGINE] Final sync error: {e}")
        
        # Cleanup
        mt5.shutdown()
        print("[ENGINE] MT5 connection closed")

    # -------------------------------------------------------------------------
    # Symbol management
    # -------------------------------------------------------------------------
    def _get_symbols_from_config(self):
        """
        Try to load symbols from:
          1) scanner.symbols
          2) top-level symbols
        """
        scanner_symbols = self.scanner_config.get("symbols")
        if scanner_symbols and isinstance(scanner_symbols, list):
            return scanner_symbols

        top_symbols = self.config.get("symbols")
        if top_symbols and isinstance(top_symbols, list):
            return top_symbols

        raise ValueError(
            "No symbols defined in master_config.yaml. "
            "Expected either 'scanner.symbols' or top-level 'symbols' list."
        )

    # -------------------------------------------------------------------------
    # Scanner main loop
    # -------------------------------------------------------------------------
    def scan_symbols(self, symbols):
        """
        Main scanning loop.
        - Pulls data for each symbol
        - Calculates indicators & price action
        - Scores BUY/SELL/NO TRADE
        - Prints breakdown ONLY for BUY/SELL signals (conservative mode)
        """
        print("=" * 80)
        print("REBEL CONSERVATIVE SCANNER - STARTING SCAN")
        print("=" * 80)

        total_signals = 0

        for symbol in symbols:
            try:
                signal, score, breakdown = self._analyze_symbol(symbol)
            except Exception as e:
                print(f"[WARN] Failed to analyze {symbol}: {e}")
                continue

            if signal in ("BUY", "SELL"):
                total_signals += 1
                self._print_signal(symbol, signal, score, breakdown)

        if total_signals == 0:
            print("No conservative BUY/SELL signals this scan (all NO TRADE).")
        else:
            print(f"Scan complete. Total conservative signals: {total_signals}")

        print("=" * 80)

    # -------------------------------------------------------------------------
    # Symbol analysis
    # -------------------------------------------------------------------------
    def _analyze_symbol(self, symbol: str):
        """
        Pulls price data for a symbol and returns:
          - signal: "BUY", "SELL", or "NO_TRADE"
          - score: numeric score (positive for buy, negative for sell)
          - breakdown: list of text lines explaining the score
        """
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Symbol {symbol} could not be selected in MT5.")

        rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, self.history_bars)
        if rates is None or len(rates) < max(self.ema_slow, self.rsi_period, self.adx_period, self.atr_period) + 5:
            raise RuntimeError(f"Not enough data for {symbol}")

        df = pd.DataFrame(rates)
        # Convert time to datetime for any future use if needed
        df["time"] = pd.to_datetime(df["time"], unit="s")

        indicators = self._compute_indicators(df)
        patterns = self._analyze_price_action(df)

        signal, score, breakdown = self._score_symbol(indicators, patterns)
        
        # ---- ML FEATURE LOGGING ----
        # Extract last values from indicator series
        ema_fast_val = float(indicators["ema_fast"].iloc[-1])
        ema_slow_val = float(indicators["ema_slow"].iloc[-1])
        rsi_val = float(indicators["rsi"].iloc[-1])
        atr_val = float(indicators["atr"].iloc[-1])
        adx_val = float(indicators["adx"].iloc[-1])
        
        # Determine trend bias
        if ema_fast_val > ema_slow_val:
            trend_bias = "bullish"
        elif ema_fast_val < ema_slow_val:
            trend_bias = "bearish"
        else:
            trend_bias = "neutral"
        
        # Determine volatility regime (compare ATR to its recent average)
        atr_series = indicators["atr"]
        atr_median = float(atr_series.median()) if len(atr_series) > 0 else atr_val
        if atr_median > 0:
            atr_ratio = atr_val / atr_median
            if atr_ratio > 1.5:
                volatility_regime = "expanding"
            elif atr_ratio < 0.7:
                volatility_regime = "contracting"
            else:
                volatility_regime = "normal"
        else:
            volatility_regime = "unknown"
        
        # Get spread ratio if possible
        spread_ratio = 0.0
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick and atr_val > 0:
                spread = tick.ask - tick.bid
                spread_ratio = spread / atr_val
        except:
            pass
        
        # Build feature dict and log
        feature_dict = {
            "ema_fast": round(ema_fast_val, 6),
            "ema_slow": round(ema_slow_val, 6),
            "rsi": round(rsi_val, 2),
            "atr": round(atr_val, 6),
            "adx": round(adx_val, 2),
            "macd_hist": None,  # Not computed in current engine
            "trend_bias": trend_bias,
            "volatility_regime": volatility_regime,
            "spread_ratio": round(spread_ratio, 4),
            "session_state": "active",  # Can be enhanced with market hours logic
            "raw_signal": signal,
            "signal_score": score,
            "reason": "; ".join(breakdown[:3]) if breakdown else "",
        }
        
        if self.ml_logger:
            self.ml_logger.log_features(symbol, feature_dict)
        
        return signal, score, breakdown

    # -------------------------------------------------------------------------
    # Indicators: EMA, RSI, ATR, ADX
    # -------------------------------------------------------------------------
    def _compute_indicators(self, df: pd.DataFrame):
        close = df["close"]

        # EMA
        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()

        # RSI
        rsi = self._compute_rsi(close, self.rsi_period)

        # ATR
        atr = self._compute_atr(df, self.atr_period)

        # ADX
        adx = self._compute_adx(df, self.adx_period)

        return {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi": rsi,
            "atr": atr,
            "adx": adx,
        }

    def _compute_rsi(self, series: pd.Series, period: int):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_atr(self, df: pd.DataFrame, period: int):
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _compute_adx(self, df: pd.DataFrame, period: int):
        # Simplified ADX calculation
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0.0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0.0)

        tr = self._compute_atr(df, period) * period  # reuse ATR logic as TR sum approx

        plus_di = 100 * (plus_dm.rolling(window=period).sum() / (tr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / (tr + 1e-10))

        dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-10)
        adx = dx.rolling(window=period).mean()
        return adx

    # -------------------------------------------------------------------------
    # Lot size calculation
    # -------------------------------------------------------------------------
    def _calc_lot(self, symbol: str, entry_price: float, sl_price: float) -> float:
        """
        Calculate lot size based on risk percentage and stop loss distance.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            sl_price: Stop loss price
            
        Returns:
            Calculated lot size
        """
        account = mt5.account_info()
        if account is None:
            raise RuntimeError("Failed to get account info from MT5")
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbol info not found for {symbol}")
        
        # 1) Risk amount in MONEY (3% by default)
        risk_percent = getattr(self, "risk_percent", 3.0)  # fallback to 3 if missing
        risk_amount = account.balance * (risk_percent / 100.0)
        
        # 2) Stop loss distance in price
        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            raise ValueError("SL distance must be greater than zero")
        
        # 3) Convert SL distance into number of ticks
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value  # money per tick at 1.0 lot
        if tick_size <= 0 or tick_value <= 0:
            raise RuntimeError(f"Invalid tick settings for {symbol}: "
                               f"tick_size={tick_size}, tick_value={tick_value}")
        
        ticks = sl_distance / tick_size
        
        # 4) Money lost per 1.0 lot if SL is hit
        loss_per_lot = ticks * tick_value
        if loss_per_lot <= 0:
            raise RuntimeError(f"Invalid loss_per_lot for {symbol}")
        
        # 5) Raw lot size for the target risk
        lot = risk_amount / loss_per_lot
        
        # 6) Normalize to broker constraints
        vol_min = symbol_info.volume_min
        vol_max = symbol_info.volume_max
        vol_step = symbol_info.volume_step
        if vol_step <= 0:
            vol_step = 0.01  # safe fallback
        
        # Round to nearest allowed step
        lot = round(lot / vol_step) * vol_step
        
        # Clamp to min/max volume
        if lot < vol_min:
            lot = vol_min
        if lot > vol_max:
            lot = vol_max
        
        return lot

    # -------------------------------------------------------------------------
    # Price action & structure
    # -------------------------------------------------------------------------
    def _analyze_price_action(self, df: pd.DataFrame):
        """
        Full price-action mode:
        - Engulfing (bull/bear)
        - Wick analysis
        - Hammer / shooting star style
        - Doji / indecision
        - Simple support / resistance proximity
        """
        # Use last two candles for pattern logic
        last = df.iloc[-1]
        prev = df.iloc[-2]

        patterns = {
            "bullish_engulf": False,
            "bearish_engulf": False,
            "hammer_like": False,
            "shooting_star_like": False,
            "doji": False,
            "near_support": False,
            "near_resistance": False,
            "bullish_rejection": False,
            "bearish_rejection": False,
        }

        # Candle metrics
        body_last = abs(last["close"] - last["open"])
        range_last = last["high"] - last["low"] + 1e-10
        upper_wick_last = last["high"] - max(last["open"], last["close"])
        lower_wick_last = min(last["open"], last["close"]) - last["low"]

        body_prev = abs(prev["close"] - prev["open"])

        # 1) Engulfing
        if body_last > body_prev * 1.1:  # body noticeably bigger
            # bullish engulf
            if last["close"] > last["open"] and prev["close"] < prev["open"]:
                if last["open"] <= prev["close"] and last["close"] >= prev["open"]:
                    patterns["bullish_engulf"] = True
            # bearish engulf
            if last["close"] < last["open"] and prev["close"] > prev["open"]:
                if last["open"] >= prev["close"] and last["close"] <= prev["open"]:
                    patterns["bearish_engulf"] = True

        # 2) Doji
        if body_last <= range_last * 0.1:
            patterns["doji"] = True

        # 3) Hammer / shooting star style (wick vs body ratios)
        if lower_wick_last >= body_last * 2 and upper_wick_last <= body_last:
            patterns["hammer_like"] = True
        if upper_wick_last >= body_last * 2 and lower_wick_last <= body_last:
            patterns["shooting_star_like"] = True

        # 4) Simple support / resistance zones using recent highs/lows
        lookback = 20
        recent = df.tail(lookback)
        recent_lows = recent["low"].min()
        recent_highs = recent["high"].max()

        # Within 10% of recent extreme
        if (last["low"] - recent_lows) / (recent_highs - recent_lows + 1e-10) <= 0.1:
            patterns["near_support"] = True
        if (recent_highs - last["high"]) / (recent_highs - recent_lows + 1e-10) <= 0.1:
            patterns["near_resistance"] = True

        # 5) Rejection context
        # Bullish rejection: long lower wick near support
        if patterns["near_support"] and lower_wick_last > body_last * 1.5:
            patterns["bullish_rejection"] = True

        # Bearish rejection: long upper wick near resistance
        if patterns["near_resistance"] and upper_wick_last > body_last * 1.5:
            patterns["bearish_rejection"] = True

        return patterns

    # -------------------------------------------------------------------------
    # Scoring & signal decision (conservative mode)
    # -------------------------------------------------------------------------
    def _score_symbol(self, indicators, patterns):
        ema_fast = indicators["ema_fast"].iloc[-1]
        ema_slow = indicators["ema_slow"].iloc[-1]
        rsi_last = indicators["rsi"].iloc[-1]
        adx_last = indicators["adx"].iloc[-1]
        atr_series = indicators["atr"]
        atr_last = atr_series.iloc[-1]
        atr_prev = atr_series.iloc[-5] if len(atr_series) > 5 else atr_series.iloc[0]

        buy_score = 0
        sell_score = 0
        breakdown = []

        # --- EMA trend ---
        if ema_fast > ema_slow:
            buy_score += 1
            breakdown.append("+1 EMA Trend (fast > slow, bullish bias)")
        elif ema_fast < ema_slow:
            sell_score -= 1
            breakdown.append("-1 EMA Trend (fast < slow, bearish bias)")

        # --- RSI zones ---
        if rsi_last > 55:
            buy_score += 1
            breakdown.append(f"+1 RSI Bullish (RSI={rsi_last:.1f} > 55)")
        elif rsi_last < 45:
            sell_score -= 1
            breakdown.append(f"-1 RSI Bearish (RSI={rsi_last:.1f} < 45)")

        # --- ADX strength ---
        if adx_last > 20:
            # boost whichever side has trend
            if buy_score > abs(sell_score):
                buy_score += 1
                breakdown.append(f"+1 ADX Trend Strength (ADX={adx_last:.1f}, supports bullish)")
            elif sell_score < -abs(buy_score):
                sell_score -= 1
                breakdown.append(f"-1 ADX Trend Strength (ADX={adx_last:.1f}, supports bearish)")
            else:
                # Neutral trend, but strong ADX – no side preference
                breakdown.append(f"0 ADX Strong but neutral bias (ADX={adx_last:.1f})")

        # --- ATR volatility trend ---
        if atr_last > atr_prev * 1.1:
            # increasing volatility
            if buy_score > abs(sell_score):
                buy_score += 1
                breakdown.append("+1 ATR Rising (volatility expansion with bullish bias)")
            elif sell_score < -abs(buy_score):
                sell_score -= 1
                breakdown.append("-1 ATR Rising (volatility expansion with bearish bias)")
            else:
                breakdown.append("0 ATR Rising (volatility up, bias unclear)")
        elif atr_last < atr_prev * 0.9:
            # contraction – penalize both sides a bit by not scoring anything extra
            breakdown.append("0 ATR Falling (volatility contraction, less attractive)")

        # --- Price-action patterns ---
        if patterns["bullish_engulf"]:
            buy_score += 2
            breakdown.append("+2 Bullish Engulfing pattern")
        if patterns["bearish_engulf"]:
            sell_score -= 2
            breakdown.append("-2 Bearish Engulfing pattern")

        if patterns["hammer_like"]:
            buy_score += 1
            breakdown.append("+1 Hammer-type candle (long lower wick)")
        if patterns["shooting_star_like"]:
            sell_score -= 1
            breakdown.append("-1 Shooting-star-type candle (long upper wick)")

        if patterns["doji"]:
            breakdown.append("0 Doji/indecision candle (context dependent)")

        if patterns["bullish_rejection"]:
            buy_score += 2
            breakdown.append("+2 Bullish rejection at/near support")
        if patterns["bearish_rejection"]:
            sell_score -= 2
            breakdown.append("-2 Bearish rejection at/near resistance")

        # ---------------------------------------------------------------------
        # Final decision (Conservative Mode)
        # ---------------------------------------------------------------------
        signal = "NO_TRADE"
        final_score = 0

        # Decide side by comparing magnitudes
        if buy_score >= self.min_score and buy_score >= abs(sell_score):
            signal = "BUY"
            final_score = buy_score
        elif abs(sell_score) >= self.min_score and abs(sell_score) > buy_score:
            signal = "SELL"
            final_score = sell_score

        # If no strong side, mark as NO_TRADE and keep breakdown unprinted
        return signal, final_score, breakdown

    # -------------------------------------------------------------------------
    # Output formatting
    # -------------------------------------------------------------------------
    def _print_signal(self, symbol, signal, score, breakdown):
        print("-" * 60)
        print(f"SYMBOL: {symbol}")
        print(f"SIGNAL: {signal}")
        print(f"SCORE: {score} (threshold={self.min_score})")
        print("")
        print("Breakdown:")
        for line in breakdown:
            print("  " + line)
        print(f"\nFinal Decision: {signal} (Conservative signal fired)")
        print("-" * 60)


if __name__ == "__main__":
    engine = RebelEngine()
    engine.start()
