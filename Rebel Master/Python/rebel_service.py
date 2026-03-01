import os
import csv
import json
import time
import subprocess
import threading
import psutil
import yaml
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

SERVICE_STATE_PATH = r"C:\Rebel Technologies\Rebel Master\Config\service_state.json"
CONFIG_PATH = r"C:\Rebel Technologies\Rebel Master\Config\master_config.yaml"
STATE_FILE = "service_state.json"
POLL_INTERVAL = 10  # seconds
ENGINE_SCRIPT_PATH = r"C:\Rebel Technologies\Rebel Master\Python\main.py"
ENGINE_PROCESS_NAME = "python.exe"  # for health check

# Performance Supervisor
LABELS_FILE = r"C:\Rebel Technologies\Rebel Master\ML\labels.csv"
PERFORMANCE_STATE_FILE = "performance_state.json"
PERFORMANCE_POLL_INTERVAL = 300  # 5 minutes

# ML Milestone Tracker
TRADE_FEATURES_FILE = r"C:\Rebel Technologies\Rebel Master\ML\trade_features.csv"
ML_MILESTONE_STATE_FILE = r"C:\Rebel Technologies\Rebel Master\Config\ml_milestone_state.json"
ML_MILESTONE_POLL_INTERVAL = 600  # 10 minutes
ML_MILESTONES = [100, 200, 300, 500, 1000]  # Notification thresholds

# Auto-Merge Settings
AUTO_MERGE_INTERVAL = 14400  # 4 hours
MERGE_SCRIPT_PATH = r"C:\Rebel Technologies\Rebel Master\ML\merge_features_labels.py"
TRAINING_DATASET_FILE = r"C:\Rebel Technologies\Rebel Master\ML\training_dataset.csv"

ENGINE_PROCESS = None


# ================================
# JSON HELPERS
# ================================
def load_service_state() -> dict:
    if not os.path.exists(SERVICE_STATE_PATH):
        return {}

    try:
        with open(SERVICE_STATE_PATH, "r") as f:
            return json.load(f)
    except:
        return {}


def save_service_state(data: dict):
    try:
        with open(SERVICE_STATE_PATH, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[SERVICE] ERROR writing state: {e}")


# ================================
# ASSET CLASSIFICATION + SNAPSHOT
# ================================
def classify_asset(symbol: str) -> str:
    s = symbol.upper()
    
    # Crypto detection
    crypto_tokens = ["BTC", "ETH", "XRP", "SOL", "ADA", "DOT", "LTC", "XLM", "DOG", 
                     "AVAX", "AAVE", "UNI", "SUSHI", "COMP", "CRV", "LRC", "MANA", 
                     "SAND", "BAT", "BNB", "KSM", "XTZ", "LNK"]
    if any(x in s for x in crypto_tokens):
        return "CRYPTO"
    
    # Metals detection
    if any(x in s for x in ["XAU", "XAG", "XPT", "GOLD", "SILVER", "COPPER"]):
        return "METALS"
    
    # Energies detection (Oil, Gas)
    if any(x in s for x in ["BRENT", "WTI", "OIL", "UKOIL", "USOIL", "NATGAS", "NGAS", "GAS", "CRUDE"]):
        return "ENERGIES"
    
    # Softs/Commodities detection
    if any(x in s for x in ["COCOA", "COFFEE", "SUGAR", "COTTON", "WHEAT", "CORN", "SOY", "BEAN"]):
        return "SOFTS"
    
    # Indices detection
    if any(x in s for x in ["US30", "US500", "US2000", "NAS", "SPX", "GER", "UK100", "UK50", 
                             "DAX", "JPN", "JP225", "HK50", "HSI", "AUS", "SPI", "CN50", 
                             "CHINA", "NETH", "IT40", "FT100", "VIX", "USTECH"]):
        return "INDICES"
    
    # Default to FX
    return "FX"


def get_board_snapshot():
    positions = mt5.positions_get()
    snapshot = {
        "open_trades": 0,
        "by_asset_class": {"FX": 0, "CRYPTO": 0, "METALS": 0, "INDICES": 0, "ENERGIES": 0, "SOFTS": 0},
        "positions": []
    }

    if positions is None:
        return snapshot

    for p in positions:
        side = "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT"
        asset_class = classify_asset(p.symbol)

        snapshot["open_trades"] += 1
        snapshot["by_asset_class"][asset_class] += 1
        snapshot["positions"].append({
            "symbol": p.symbol,
            "side": side,
            "volume": p.volume
        })

    return snapshot


# ================================
# HEALTH SNAPSHOT
# ================================
def get_health_snapshot(mt5_ok: bool, poll_drift: float | None, last_mt5_ok_time):
    engine_alive = any(
        p.info["name"] == ENGINE_PROCESS_NAME
        for p in psutil.process_iter(["name"])
    )

    return {
        "supervisor_alive": True,
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "poll_drift_seconds": poll_drift,
        "mt5_connected": mt5_ok,
        "last_mt5_ok": last_mt5_ok_time.isoformat() if last_mt5_ok_time else None,
        "engine_alive": engine_alive
    }


# ================================
# BOARD SNAPSHOT LOOP
# ================================
def board_snapshot_loop():
    last_snapshot = None
    last_heartbeat = None
    last_mt5_ok = None
    expected_interval = POLL_INTERVAL
    consecutive_failures = 0
    MAX_FAILURES_BEFORE_RECONNECT = 3

    while True:
        now = datetime.now(timezone.utc)

        # Poll drift calculation
        poll_drift = None
        if last_heartbeat is not None:
            actual_interval = (now - last_heartbeat).total_seconds()
            poll_drift = round(actual_interval - expected_interval, 3)

        last_heartbeat = now

        # MT5 health check with reconnection
        mt5_ok = mt5.positions_get() is not None
        
        if mt5_ok:
            last_mt5_ok = now
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            print(f"[SERVICE] MT5 check failed ({consecutive_failures}/{MAX_FAILURES_BEFORE_RECONNECT})", flush=True)
            
            # Try to reconnect after multiple failures
            if consecutive_failures >= MAX_FAILURES_BEFORE_RECONNECT:
                print("[SERVICE] Attempting MT5 reconnection...", flush=True)
                try:
                    # Load config for path
                    with open(CONFIG_PATH, 'r') as f:
                        config = yaml.safe_load(f)
                        mt5_path = config.get("mt5", {}).get("path")
                    
                    if connect_mt5_with_retry(mt5_path, max_retries=3, retry_delay=5):
                        consecutive_failures = 0
                        print("[SERVICE] MT5 reconnected successfully", flush=True)
                except Exception as e:
                    print(f"[SERVICE] Reconnection error: {e}", flush=True)

        snapshot = get_board_snapshot()

        if snapshot != last_snapshot:
            snapshot["timestamp"] = now.isoformat()
            snapshot["health"] = get_health_snapshot(mt5_ok, poll_drift, last_mt5_ok)

            with open(STATE_FILE, "w") as f:
                json.dump(snapshot, f, indent=2)

            print(f"[SERVICE] Board updated | Open trades: {snapshot['open_trades']}", flush=True)
            last_snapshot = snapshot

        time.sleep(POLL_INTERVAL)


# ================================
# ENGINE CONTROL
# ================================
def start_engine():
    global ENGINE_PROCESS

    if ENGINE_PROCESS is not None and ENGINE_PROCESS.poll() is None:
        print("[SERVICE] Engine already running.")
        return

    print("[SERVICE] Starting engine...")
    try:
        ENGINE_PROCESS = subprocess.Popen(
            ["python", ENGINE_SCRIPT_PATH],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        save_service_state({
            "state": "running",
            "command": "",
            "engine_pid": ENGINE_PROCESS.pid,
            "health": load_service_state().get("health", {})
        })
    except Exception as e:
        print(f"[SERVICE] ERROR starting engine: {e}")
        save_service_state({"state": "crashed", "command": ""})


def stop_engine():
    global ENGINE_PROCESS

    print("[SERVICE] Stopping engine...")
    try:
        if ENGINE_PROCESS is not None and ENGINE_PROCESS.poll() is None:
            ENGINE_PROCESS.terminate()
            time.sleep(1)
    except Exception as e:
        print(f"[SERVICE] ERROR stopping engine: {e}")

    ENGINE_PROCESS = None
    save_service_state({
        "state": "stopped",
        "command": "",
        "engine_pid": None,
        "health": load_service_state().get("health", {})
    })


# ================================
# HEALTH MONITOR
# ================================
def get_system_health(engine_pid: int):
    cpu_total = psutil.cpu_percent(interval=0.2)
    ram_total = psutil.virtual_memory().percent

    engine_cpu = 0
    engine_ram = 0

    if engine_pid:
        try:
            p = psutil.Process(engine_pid)
            engine_cpu = p.cpu_percent(interval=0.1)
            engine_ram = p.memory_info().rss / (1024 * 1024)
        except:
            pass

    return {
        "cpu_total": round(cpu_total, 2),
        "ram_total": round(ram_total, 2),
        "engine_cpu": round(engine_cpu, 2),
        "engine_ram_mb": round(engine_ram, 2),
        "updated": time.time()
    }


def update_health_state():
    while True:
        state = load_service_state()
        engine_pid = state.get("engine_pid", None)

        health = get_system_health(engine_pid)
        state["health"] = health

        save_service_state(state)
        time.sleep(3)


# ================================
# SUPERVISOR LOOP — FIXED
# ================================
def supervisor_loop():
    print("[SERVICE] Supervisor running...")

    while True:
        try:
            state = load_service_state()
            command = state.get("command", "").lower()

            if command == "start":
                print("[SERVICE] Command received: START")
                state = load_service_state()
                state["state"] = "starting"
                state["command"] = ""
                save_service_state(state)
                start_engine()

            elif command == "stop":
                print("[SERVICE] Command received: STOP")
                state = load_service_state()
                state["state"] = "stopping"
                state["command"] = ""
                save_service_state(state)
                stop_engine()

            elif command == "restart":
                print("[SERVICE] Command received: RESTART")
                state = load_service_state()
                state["state"] = "restarting"
                state["command"] = ""
                save_service_state(state)
                stop_engine()
                time.sleep(2)
                start_engine()

        except Exception as e:
            print(f"[SERVICE] ERROR in supervisor loop: {e}")

        time.sleep(1)


# ================================
# PERFORMANCE SUPERVISOR
# ================================
def get_rolling_window_bounds():
    """
    Calculate Sunday 14:00 → Friday 14:00 (local time) window bounds.
    Returns (window_start, window_end) as datetime objects.
    """
    now = datetime.now()
    weekday = now.weekday()  # Monday=0, Sunday=6

    # Find last Sunday 14:00
    days_since_sunday = (weekday + 1) % 7
    last_sunday = now - timedelta(days=days_since_sunday)
    window_start = last_sunday.replace(hour=14, minute=0, second=0, microsecond=0)

    # If we're before Sunday 14:00, go back another week
    if now < window_start:
        window_start -= timedelta(days=7)

    # Friday 14:00 is 5 days after Sunday
    window_end = window_start + timedelta(days=5)

    return window_start, window_end


def compute_stats(trades: list) -> dict:
    """
    Compute win/loss stats from a list of trade dicts.
    Each trade has 'symbol', 'outcome_class' ('win' or 'loss'), and optional 'rr'.
    RR is skipped if missing/invalid (backward compatible with old data).
    """
    total = len(trades)
    wins = sum(1 for t in trades if t["outcome_class"] == "win")
    losses = total - wins
    win_rate = round((wins / total) * 100, 1) if total > 0 else 0.0

    # Overall average RR (only trades with valid RR)
    rr_values = [t["rr"] for t in trades if t.get("rr") is not None]
    avg_rr = round(sum(rr_values) / len(rr_values), 2) if rr_values else None

    # By asset class
    by_class = {"FX": [], "CRYPTO": [], "METALS": [], "INDICES": [], "ENERGIES": [], "SOFTS": []}
    for t in trades:
        asset_class = classify_asset(t["symbol"])
        if asset_class not in by_class:
            by_class[asset_class] = []
        by_class[asset_class].append(t)

    by_asset_class = {}
    for ac, ac_trades in by_class.items():
        ac_total = len(ac_trades)
        ac_wins = sum(1 for t in ac_trades if t["outcome_class"] == "win")
        ac_win_rate = round((ac_wins / ac_total) * 100, 1) if ac_total > 0 else 0.0
        ac_rr_values = [t["rr"] for t in ac_trades if t.get("rr") is not None]
        ac_avg_rr = round(sum(ac_rr_values) / len(ac_rr_values), 2) if ac_rr_values else None
        by_asset_class[ac] = {
            "trades": ac_total,
            "wins": ac_wins,
            "win_rate": ac_win_rate,
            "avg_rr": ac_avg_rr
        }

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "by_asset_class": by_asset_class
    }


def read_labels() -> list:
    """
    Read all labels from ML/labels.csv.
    Returns list of dicts with 'timestamp', 'symbol', 'outcome_class', 'rr'.
    RR is None if missing/invalid (backward compatible).
    """
    trades = []
    if not os.path.exists(LABELS_FILE):
        return trades

    try:
        with open(LABELS_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Parse timestamp (format: "YYYY-MM-DD HH:MM:SS")
                    ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    
                    # Parse RR (None if missing or invalid)
                    rr = None
                    rr_str = row.get("rr", "").strip()
                    if rr_str:
                        try:
                            rr = float(rr_str)
                        except ValueError:
                            pass
                    
                    trades.append({
                        "timestamp": ts,
                        "symbol": row["symbol"],
                        "outcome_class": row["outcome_class"],
                        "rr": rr
                    })
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"[PERF] Error reading labels: {e}", flush=True)

    return trades


def performance_supervisor_loop():
    """
    Performance Supervisor - runs every 120 seconds.
    Reads ML labels and computes lifetime + rolling 5-day stats.
    """
    last_stats = None

    print("[PERF] Performance Supervisor started.", flush=True)

    while True:
        try:
            trades = read_labels()
            window_start, window_end = get_rolling_window_bounds()

            # Filter trades for rolling window
            rolling_trades = [
                t for t in trades
                if window_start <= t["timestamp"] <= window_end
            ]

            # Compute stats
            lifetime_stats = compute_stats(trades)
            rolling_stats = compute_stats(rolling_trades)
            rolling_stats["window"] = "Sun 14:00 → Fri 14:00"

            # Build output
            stats = {
                "lifetime": lifetime_stats,
                "rolling_5_day": rolling_stats
            }

            # Only write if changed
            if stats != last_stats:
                output = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **stats
                }

                with open(PERFORMANCE_STATE_FILE, "w") as f:
                    json.dump(output, f, indent=2)

                # Full stats display
                print("\n" + "=" * 60, flush=True)
                print("[PERF] PERFORMANCE UPDATE", flush=True)
                print("=" * 60, flush=True)
                
                # Lifetime stats
                lt = lifetime_stats
                print(f"\n📊 LIFETIME STATS", flush=True)
                print(f"   Total: {lt['total_trades']} trades | Wins: {lt['wins']} | Losses: {lt['losses']}", flush=True)
                print(f"   Win Rate: {lt['win_rate']}% | Avg RR: {lt['avg_rr'] or 'N/A'}", flush=True)
                print(f"   By Asset Class:", flush=True)
                for ac, ac_stats in lt['by_asset_class'].items():
                    if ac_stats['trades'] > 0:
                        print(f"      {ac}: {ac_stats['trades']} trades, {ac_stats['win_rate']}% WR, RR: {ac_stats['avg_rr'] or 'N/A'}", flush=True)
                
                # Rolling 5-day stats
                rs = rolling_stats
                print(f"\n📈 ROLLING 5-DAY ({rs['window']})", flush=True)
                print(f"   Total: {rs['total_trades']} trades | Wins: {rs['wins']} | Losses: {rs['losses']}", flush=True)
                print(f"   Win Rate: {rs['win_rate']}% | Avg RR: {rs['avg_rr'] or 'N/A'}", flush=True)
                print(f"   By Asset Class:", flush=True)
                for ac, ac_stats in rs['by_asset_class'].items():
                    if ac_stats['trades'] > 0:
                        print(f"      {ac}: {ac_stats['trades']} trades, {ac_stats['win_rate']}% WR, RR: {ac_stats['avg_rr'] or 'N/A'}", flush=True)
                
                print("\n" + "=" * 60 + "\n", flush=True)
                last_stats = stats

        except Exception as e:
            print(f"[PERF] Error in performance loop: {e}", flush=True)

        time.sleep(PERFORMANCE_POLL_INTERVAL)


# ================================
# ML MILESTONE TRACKER
# ================================
def load_ml_milestone_state() -> dict:
    """Load ML milestone state from JSON file."""
    if not os.path.exists(ML_MILESTONE_STATE_FILE):
        return {"last_notified_milestone": 0}
    try:
        with open(ML_MILESTONE_STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {"last_notified_milestone": 0}


def save_ml_milestone_state(data: dict):
    """Save ML milestone state to JSON file."""
    try:
        os.makedirs(os.path.dirname(ML_MILESTONE_STATE_FILE), exist_ok=True)
        with open(ML_MILESTONE_STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[ML_MILESTONE] Error saving state: {e}", flush=True)


def count_matched_trades() -> tuple:
    """
    Count matched trades from training_dataset.csv (the merged dataset).
    Returns (matched_count, wins, losses, win_rate).
    
    Uses training_dataset.csv which is created by merge_features_labels.py
    and represents the actual number of trades ready for ML training.
    """
    training_dataset_file = r"C:\Rebel Technologies\Rebel Master\ML\training_dataset.csv"
    
    # Try to read training_dataset.csv first (most accurate)
    if os.path.exists(training_dataset_file):
        try:
            matched_count = 0
            wins = 0
            losses = 0
            
            with open(training_dataset_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    matched_count += 1
                    try:
                        label = int(row.get("label", 0))
                        if label == 1:
                            wins += 1
                        else:
                            losses += 1
                    except (ValueError, KeyError):
                        losses += 1  # Default to loss if label missing
            
            win_rate = round((wins / matched_count) * 100, 1) if matched_count > 0 else 0.0
            return matched_count, wins, losses, win_rate
            
        except Exception as e:
            print(f"[ML_MILESTONE] Error reading training_dataset: {e}", flush=True)
    
    # Fallback: count from raw files if training_dataset doesn't exist
    feature_tickets = set()
    label_data = {}  # ticket -> label (0 or 1)
    
    # Read trade features
    if os.path.exists(TRADE_FEATURES_FILE):
        try:
            with open(TRADE_FEATURES_FILE, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        feature_tickets.add(int(row["ticket"]))
                    except (ValueError, KeyError):
                        pass
        except Exception as e:
            print(f"[ML_MILESTONE] Error reading trade_features: {e}", flush=True)
    
    # Read labels
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ticket = int(row["ticket"])
                        label = int(row["label"])
                        label_data[ticket] = label
                    except (ValueError, KeyError):
                        pass
        except Exception as e:
            print(f"[ML_MILESTONE] Error reading labels: {e}", flush=True)
    
    # Count matches
    matched_tickets = feature_tickets & set(label_data.keys())
    matched_count = len(matched_tickets)
    
    wins = sum(1 for t in matched_tickets if label_data.get(t, 0) == 1)
    losses = matched_count - wins
    win_rate = round((wins / matched_count) * 100, 1) if matched_count > 0 else 0.0
    
    return matched_count, wins, losses, win_rate


def send_ml_milestone_telegram(milestone: int, matched: int, wins: int, losses: int, win_rate: float) -> bool:
    """Send Telegram notification for ML milestone."""
    try:
        # Load config to get Telegram settings
        config_path = r"C:\Rebel Technologies\Rebel Master\Config\master_config.yaml"
        if not os.path.exists(config_path):
            print("[ML_MILESTONE] Config not found, skipping Telegram", flush=True)
            return False
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        telegram_cfg = config.get("telegram", {})
        if not telegram_cfg.get("enabled", False):
            print("[ML_MILESTONE] Telegram not enabled", flush=True)
            return False
        
        bot_token = telegram_cfg.get("bot_token", "")
        chat_id = telegram_cfg.get("chat_id", "")
        
        if not bot_token or not chat_id:
            return False
        
        # Build message
        next_milestone = None
        for m in ML_MILESTONES:
            if m > milestone:
                next_milestone = m
                break
        
        text = (
            f"🎯 <b>ML MILESTONE REACHED!</b>\n\n"
            f"📊 Matched Trades: <b>{matched}</b>\n"
            f"✅ Wins: {wins} | ❌ Losses: {losses}\n"
            f"📈 Win Rate: {win_rate}%\n\n"
        )
        
        if milestone >= 300:
            text += "🚀 <b>RF v3 READY!</b> You have enough data for robust training.\n"
        elif milestone >= 100:
            text += "⚡ <b>RF v3 MINIMUM MET!</b> Can start training (more data = better).\n"
        
        if next_milestone:
            text += f"\n🎯 Next milestone: {next_milestone} trades"
        
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
        
    except Exception as e:
        print(f"[ML_MILESTONE] Telegram error: {e}", flush=True)
        return False


def ml_milestone_tracker_loop():
    """
    ML Milestone Tracker - runs every 10 minutes.
    Sends Telegram notification when matched trades hit milestones.
    """
    print("[ML_MILESTONE] ML Milestone Tracker started.", flush=True)
    
    while True:
        try:
            matched, wins, losses, win_rate = count_matched_trades()
            state = load_ml_milestone_state()
            last_notified = state.get("last_notified_milestone", 0)
            
            # Check if we've hit a new milestone
            for milestone in ML_MILESTONES:
                if matched >= milestone and milestone > last_notified:
                    print(f"[ML_MILESTONE] 🎯 Milestone reached: {milestone} trades!", flush=True)
                    
                    # Send notification
                    if send_ml_milestone_telegram(milestone, matched, wins, losses, win_rate):
                        print(f"[ML_MILESTONE] Telegram notification sent", flush=True)
                    
                    # Update state
                    state["last_notified_milestone"] = milestone
                    state["last_check"] = datetime.now(timezone.utc).isoformat()
                    state["current_matched"] = matched
                    save_ml_milestone_state(state)
                    break  # Only notify one milestone per check
            
            # Log current status periodically
            print(f"[ML_MILESTONE] Current: {matched} matched trades (W:{wins}/L:{losses}, {win_rate}% WR)", flush=True)
            
        except Exception as e:
            print(f"[ML_MILESTONE] Error in tracker loop: {e}", flush=True)
        
        time.sleep(ML_MILESTONE_POLL_INTERVAL)


# ================================
# AUTO-MERGE LOOP
# ================================
def run_auto_merge():
    """
    Run the merge script and clean duplicates from labels.csv.
    Called automatically every 4 hours.
    """
    try:
        print("\n" + "=" * 60, flush=True)
        print("[AUTO-MERGE] Starting automatic merge...", flush=True)
        print("=" * 60, flush=True)
        
        # Step 1: Clean duplicates from labels.csv first
        if os.path.exists(LABELS_FILE):
            try:
                import pandas as pd
                df = pd.read_csv(LABELS_FILE)
                before = len(df)
                df = df.drop_duplicates(subset='ticket', keep='first')
                after = len(df)
                if before != after:
                    df.to_csv(LABELS_FILE, index=False)
                    print(f"[AUTO-MERGE] Cleaned labels.csv: {before} -> {after} ({before - after} duplicates)", flush=True)
            except Exception as e:
                print(f"[AUTO-MERGE] Warning - could not clean labels: {e}", flush=True)
        
        # Step 2: Run merge script
        result = subprocess.run(
            ["python", MERGE_SCRIPT_PATH],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            # Extract matched count from output
            for line in result.stdout.split('\n'):
                if 'matched' in line.lower() or 'rows' in line.lower():
                    print(f"[AUTO-MERGE] {line.strip()}", flush=True)
            print("[AUTO-MERGE] Merge completed successfully", flush=True)
        else:
            print(f"[AUTO-MERGE] Merge script error: {result.stderr}", flush=True)
        
        # Step 3: Clean duplicates from training dataset
        if os.path.exists(TRAINING_DATASET_FILE):
            try:
                import pandas as pd
                df = pd.read_csv(TRAINING_DATASET_FILE)
                before = len(df)
                df = df.drop_duplicates(subset='ticket', keep='first')
                after = len(df)
                if before != after:
                    df.to_csv(TRAINING_DATASET_FILE, index=False)
                    print(f"[AUTO-MERGE] Cleaned training_dataset: {before} -> {after}", flush=True)
                print(f"[AUTO-MERGE] Training dataset: {after} unique trades", flush=True)
            except Exception as e:
                print(f"[AUTO-MERGE] Warning - could not clean dataset: {e}", flush=True)
        
        print("=" * 60 + "\n", flush=True)
        return True
        
    except subprocess.TimeoutExpired:
        print("[AUTO-MERGE] Merge script timed out", flush=True)
        return False
    except Exception as e:
        print(f"[AUTO-MERGE] Error: {e}", flush=True)
        return False


def auto_merge_loop():
    """
    Auto-merge loop - runs every 4 hours.
    Automatically merges features and labels into training dataset.
    """
    print("[AUTO-MERGE] Auto-merge loop started (runs every 4 hours)", flush=True)
    
    # Wait 5 minutes before first run (let system stabilize)
    time.sleep(300)
    
    while True:
        try:
            run_auto_merge()
        except Exception as e:
            print(f"[AUTO-MERGE] Loop error: {e}", flush=True)
        
        time.sleep(AUTO_MERGE_INTERVAL)


# ================================
# MT5 CONNECTION WITH RETRY
# ================================
def connect_mt5_with_retry(mt5_path: str, max_retries: int = 5, retry_delay: int = 3) -> bool:
    """
    Connect to MT5 with retry logic to handle platform resets.
    
    Args:
        mt5_path: Path to MT5 terminal executable
        max_retries: Number of connection attempts
        retry_delay: Seconds to wait between retries
        
    Returns:
        True if connected successfully
    """
    for attempt in range(1, max_retries + 1):
        print(f"[SERVICE] MT5 connection attempt {attempt}/{max_retries}...", flush=True)
        
        # Shutdown any existing connection
        try:
            mt5.shutdown()
        except:
            pass
        
        time.sleep(1)
        
        # Initialize
        print(f"[SERVICE] Connecting to MT5: {mt5_path}", flush=True)
        if not mt5.initialize(path=mt5_path):
            error = mt5.last_error()
            print(f"[SERVICE] MT5 initialization failed: {error}", flush=True)
            if attempt < max_retries:
                print(f"[SERVICE] Waiting {retry_delay}s before retry...", flush=True)
                time.sleep(retry_delay)
            continue
        
        # Wait for terminal to fully load (handles "resetting platform" state)
        time.sleep(2)
        
        # Verify terminal info with retries
        terminal_ready = False
        for check in range(3):
            terminal_info = mt5.terminal_info()
            if terminal_info is not None:
                terminal_ready = True
                break
            print(f"[SERVICE] Waiting for terminal info... ({check+1}/3)", flush=True)
            time.sleep(2)
        
        if not terminal_ready:
            print("[SERVICE] Failed to get terminal info - platform may be resetting", flush=True)
            if attempt < max_retries:
                print(f"[SERVICE] Waiting {retry_delay}s before retry...", flush=True)
                time.sleep(retry_delay)
            continue
        
        # Success!
        account = mt5.account_info()
        if account:
            print(f"[SERVICE] MT5 connected on attempt {attempt} - Account {account.login}", flush=True)
        else:
            print(f"[SERVICE] MT5 connected on attempt {attempt} (read-only supervisor)", flush=True)
        return True
    
    print(f"[SERVICE] Failed to connect to MT5 after {max_retries} attempts", flush=True)
    return False


# ================================
# MAIN ENTRY
# ================================
if __name__ == "__main__":
    print("[SERVICE] REBEL Supervisor started.")
    print("[SERVICE] This window must stay open unless installed as a Windows Service.")

    # Load config for MT5 path
    mt5_path = None
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            mt5_path = config.get("mt5", {}).get("path")
    except Exception as e:
        print(f"[SERVICE] Could not load config: {e}", flush=True)
    
    # Initialize MT5 with retry logic (handles platform resets)
    if not connect_mt5_with_retry(mt5_path):
        print("[SERVICE] WARNING: Running without MT5 connection", flush=True)

    # Start health monitor
    threading.Thread(target=update_health_state, daemon=True).start()

    # Start board snapshot monitor
    threading.Thread(target=board_snapshot_loop, daemon=True).start()

    # Start performance supervisor
    threading.Thread(target=performance_supervisor_loop, daemon=True).start()

    # Start ML milestone tracker
    threading.Thread(target=ml_milestone_tracker_loop, daemon=True).start()

    # Start auto-merge loop
    threading.Thread(target=auto_merge_loop, daemon=True).start()

    # Start supervisor
    supervisor_loop()
