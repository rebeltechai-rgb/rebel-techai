import json
import os
import subprocess
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import MetaTrader5 as mt5
import yaml

from rebel_service_client import send_command

DASHBOARD_CONFIG_PATH = r"C:\Rebel Technologies\Rebel Master\Config\dashboard_config.json"
RUNTIME_STATE_PATH = r"C:\Rebel Technologies\Rebel Master\State\runtime_state.json"
SYMBOL_LIST_PATH = r"C:\Rebel Technologies\Rebel Master\Config\symbol_lists.yaml"
BOT_SCRIPT_PATH = r"C:\Rebel Technologies\Rebel Master\Python\main.py"
SERVICE_STATE_FILE = r"C:\Rebel Technologies\Rebel Master\Config\service_state.json"


class RebelControlCenter:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("REBEL Command Center")
        self.window.geometry("760x560")
        self.window.configure(bg="#111")

        self.settings = self.load_dashboard_settings()
        self.bot_process = None
        self.mt5_ok = False

        self._init_mt5()
        self._build_ui()
        self.init_status_bar()
        self._start_refresh_loop()

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()

    # -----------------------------
    # CONFIG & STATE FILE HANDLING
    # -----------------------------
    def load_dashboard_settings(self):
        if not os.path.exists(DASHBOARD_CONFIG_PATH):
            return {
                "auto_trade": False,
                "strategy_mode": "normal",
                "min_score": 3,
                "timeframe": "M15",
                "add_symbols": [],
                "kill_switch": False
            }
        try:
            with open(DASHBOARD_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "auto_trade": False,
                "strategy_mode": "normal",
                "min_score": 3,
                "timeframe": "M15",
                "add_symbols": [],
                "kill_switch": False
            }

    def save_dashboard_settings(self):
        try:
            with open(DASHBOARD_CONFIG_PATH, "w") as f:
                json.dump(self.settings, f, indent=4)
            print("[GUI] Saved dashboard_config.json")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dashboard settings:\n{e}")

    def load_runtime_state(self):
        if not os.path.exists(RUNTIME_STATE_PATH):
            return {}
        try:
            with open(RUNTIME_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def load_symbols_from_yaml(self):
        if not os.path.exists(SYMBOL_LIST_PATH):
            return []
        try:
            with open(SYMBOL_LIST_PATH, "r") as f:
                data = yaml.safe_load(f)
            groups = data.get("groups", {})
            all_syms = []
            for g, lst in groups.items():
                if isinstance(lst, list):
                    all_syms.extend(lst)
            # unique, keep short for the price panel
            return list(dict.fromkeys(all_syms))[:15]
        except Exception:
            return []

    # -----------------------------
    # MT5 HANDLING (READ ONLY)
    # -----------------------------
    def _init_mt5(self):
        try:
            if not mt5.initialize():
                print("[GUI:MT5] Failed to initialize MT5")
                self.mt5_ok = False
            else:
                print("[GUI:MT5] Connected (read-only)")
                self.mt5_ok = True
        except Exception as e:
            print(f"[GUI:MT5] Exception on initialize: {e}")
            self.mt5_ok = False

    def _shutdown_mt5(self):
        try:
            if self.mt5_ok:
                mt5.shutdown()
        except Exception:
            pass

    def get_open_positions(self):
        if not self.mt5_ok:
            return []
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            result = []
            for p in positions:
                direction = "BUY" if p.type == 0 else "SELL"
                result.append({
                    "symbol": p.symbol,
                    "direction": direction,
                    "volume": p.volume,
                    "price_open": p.price_open,
                    "sl": p.sl,
                    "tp": p.tp,
                    "profit": p.profit
                })
            return result
        except Exception:
            return []

    def get_symbol_prices(self, symbols):
        if not self.mt5_ok or not symbols:
            return []
        out = []
        for sym in symbols:
            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                continue
            spread = None
            info = mt5.symbol_info(sym)
            if info is not None and info.point:
                spread = (tick.ask - tick.bid) / info.point
            out.append({
                "symbol": sym,
                "bid": tick.bid,
                "ask": tick.ask,
                "spread": spread
            })
        return out

    # -----------------------------
    # BOT CONTROL (START / STOP)
    # -----------------------------
    def is_bot_running(self):
        return self.bot_process is not None and self.bot_process.poll() is None

    def start_bot(self):
        try:
            send_command("start")
            print("[CONTROL_CENTER] Sent START command to supervisor.")
        except Exception as e:
            print(f"[CONTROL_CENTER] Failed to send START command: {e}")

    def stop_bot(self):
        try:
            send_command("stop")
            print("[CONTROL_CENTER] Sent STOP command to supervisor.")
        except Exception as e:
            print(f"[CONTROL_CENTER] Failed to send STOP command: {e}")

    def restart_bot(self):
        try:
            send_command("restart")
            print("[CONTROL_CENTER] Sent RESTART command to supervisor.")
        except Exception as e:
            print(f"[CONTROL_CENTER] Failed to send RESTART command: {e}")

    # -----------------------------
    # UI BUILD
    # -----------------------------
    def _build_ui(self):
        # TOP BAR: Title + Bot Status
        top_frame = tk.Frame(self.window, bg="#111")
        top_frame.pack(fill="x", pady=10)

        title = tk.Label(
            top_frame,
            text="REBEL COMMAND CENTER",
            bg="#111",
            fg="#00FF88",
            font=("Arial", 18, "bold")
        )
        title.pack(side="left", padx=10)

        self.status_label_var = tk.StringVar(
            value="BOT STATUS: RUNNING" if self.is_bot_running() else "BOT STATUS: STOPPED"
        )
        status_label = tk.Label(
            top_frame,
            textvariable=self.status_label_var,
            bg="#111",
            fg="#CCCCCC",
            font=("Arial", 10, "bold")
        )
        status_label.pack(side="right", padx=10)

        # CONTROLS FRAME (left)
        controls = tk.Frame(self.window, bg="#111")
        controls.pack(side="left", fill="y", padx=10, pady=5)

        # Bot control buttons
        tk.Label(controls, text="Engine Control", bg="#111", fg="#FFFFFF",
                 font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))

        tk.Button(
            controls,
            text="▶ START BOT",
            command=self.start_bot,
            bg="#006600",
            fg="white",
            width=18
        ).pack(pady=2)

        tk.Button(
            controls,
            text="■ STOP BOT",
            command=self.stop_bot,
            bg="#660000",
            fg="white",
            width=18
        ).pack(pady=2)

        # Separator
        tk.Frame(controls, height=2, bg="#333").pack(fill="x", pady=10)

        # Strategy Controls
        tk.Label(controls, text="Strategy", bg="#111", fg="#FFFFFF",
                 font=("Arial", 12, "bold")).pack(anchor="w")

        # Auto-trade
        self.auto_var = tk.BooleanVar(value=self.settings.get("auto_trade", False))
        tk.Checkbutton(
            controls,
            text="Enable Auto-Trading",
            variable=self.auto_var,
            bg="#111",
            fg="white",
            selectcolor="#222",
            command=self._on_auto_change
        ).pack(anchor="w", pady=3)

        # Mode
        tk.Label(controls, text="Mode:", bg="#111", fg="#FFFFFF").pack(anchor="w")
        self.mode_var = tk.StringVar(value=self.settings.get("strategy_mode", "normal"))
        ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=["conservative", "normal", "aggressive"],
            width=16
        ).pack(anchor="w", pady=3)

        # Min score
        score_frame = tk.Frame(controls, bg="#111")
        score_frame.pack(anchor="w", pady=3)
        tk.Label(score_frame, text="Min Score:", bg="#111", fg="#FFFFFF").pack(side="left")
        self.score_entry = tk.Entry(score_frame, width=4)
        self.score_entry.insert(0, str(self.settings.get("min_score", 3)))
        self.score_entry.pack(side="left", padx=3)

        # Timeframe
        tk.Label(controls, text="Timeframe:", bg="#111", fg="#FFFFFF").pack(anchor="w")
        self.tf_var = tk.StringVar(value=self.settings.get("timeframe", "M15"))
        ttk.Combobox(
            controls,
            textvariable=self.tf_var,
            values=["M1", "M5", "M15", "M30", "H1", "H4"],
            width=16
        ).pack(anchor="w", pady=3)

        # Add symbol
        tk.Label(controls, text="Add Symbol:", bg="#111", fg="#FFFFFF").pack(anchor="w", pady=(8, 0))
        self.symbol_entry = tk.Entry(controls, width=14)
        self.symbol_entry.pack(anchor="w", pady=3)

        tk.Button(
            controls,
            text="Add Symbol",
            command=self._on_add_symbol,
            bg="#333333",
            fg="white",
            width=18
        ).pack(pady=3)

        # Kill switch
        tk.Button(
            controls,
            text="⚠ EMERGENCY STOP",
            command=self._on_kill_switch,
            bg="#880000",
            fg="white",
            width=18,
            font=("Arial", 10, "bold")
        ).pack(pady=10)

        # Save settings
        tk.Button(
            controls,
            text="Save Settings",
            command=self._on_save_settings,
            bg="#0055AA",
            fg="white",
            width=18
        ).pack(pady=3)

        # RIGHT SIDE: ACCOUNT / POSITIONS / PRICES
        right = tk.Frame(self.window, bg="#111")
        right.pack(side="right", fill="both", expand=True, padx=10, pady=5)

        # Account panel
        acc_frame = tk.LabelFrame(right, text="Account", bg="#111", fg="#FFFFFF")
        acc_frame.pack(fill="x", pady=5)

        self.balance_var = tk.StringVar(value="Balance: N/A")
        self.equity_var = tk.StringVar(value="Equity: N/A")
        self.daily_pl_var = tk.StringVar(value="Daily P/L: N/A")

        tk.Label(acc_frame, textvariable=self.balance_var, bg="#111", fg="#FFFFFF").pack(anchor="w")
        tk.Label(acc_frame, textvariable=self.equity_var, bg="#111", fg="#FFFFFF").pack(anchor="w")
        tk.Label(acc_frame, textvariable=self.daily_pl_var, bg="#111", fg="#00FF88").pack(anchor="w")

        # Positions panel
        pos_frame = tk.LabelFrame(right, text="Open Positions", bg="#111", fg="#FFFFFF")
        pos_frame.pack(fill="both", expand=True, pady=5)

        columns = ("symbol", "direction", "volume", "price_open", "sl", "tp", "profit")
        self.pos_tree = ttk.Treeview(pos_frame, columns=columns, show="headings", height=8)
        for col in columns:
            self.pos_tree.heading(col, text=col.upper())
            self.pos_tree.column(col, width=80, anchor="center")
        self.pos_tree.pack(fill="both", expand=True)

        # Prices panel
        price_frame = tk.LabelFrame(right, text="Prices", bg="#111", fg="#FFFFFF")
        price_frame.pack(fill="x", pady=5)

        self.price_list = tk.Listbox(price_frame, height=6, bg="#111", fg="#FFFFFF")
        self.price_list.pack(fill="x")

    # -----------------------------
    # UI CALLBACKS
    # -----------------------------
    def _on_auto_change(self):
        self.settings["auto_trade"] = self.auto_var.get()

    def _on_add_symbol(self):
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            return
        self.settings.setdefault("add_symbols", [])
        if symbol not in self.settings["add_symbols"]:
            self.settings["add_symbols"].append(symbol)
        self.symbol_entry.delete(0, tk.END)
        messagebox.showinfo("Symbol Added", f"{symbol} added to dashboard overrides.")
        self.save_dashboard_settings()

    def _on_kill_switch(self):
        self.settings["kill_switch"] = True
        self.save_dashboard_settings()
        messagebox.showwarning("EMERGENCY STOP", "Kill switch activated. New trades should be blocked.")
        # also stop bot process if running
        if self.is_bot_running():
            try:
                self.bot_process.terminate()
                self.bot_process = None
                self.status_label_var.set("BOT STATUS: STOPPED (KILL)")
            except Exception:
                pass

    def _on_save_settings(self):
        try:
            self.settings["strategy_mode"] = self.mode_var.get()
            self.settings["min_score"] = int(self.score_entry.get())
            self.settings["timeframe"] = self.tf_var.get()
            self.settings["auto_trade"] = self.auto_var.get()
            self.save_dashboard_settings()
            messagebox.showinfo("Saved", "Settings saved to dashboard_config.json.")
        except ValueError:
            messagebox.showerror("Error", "Min score must be a number.")

    # -----------------------------
    # REFRESH LOOP
    # -----------------------------
    def _refresh(self):
        # Bot status
        if self.is_bot_running():
            self.status_label_var.set("BOT STATUS: RUNNING")
        else:
            self.status_label_var.set("BOT STATUS: STOPPED")

        # Runtime state (balance, equity, daily P/L)
        state = self.load_runtime_state()
        bal = state.get("balance")
        eq = state.get("equity")
        daily_pl = state.get("daily_pl")
        daily_pl_pct = state.get("daily_pl_pct")

        if bal is not None:
            self.balance_var.set(f"Balance: {bal:.2f}")
        else:
            self.balance_var.set("Balance: N/A")

        if eq is not None:
            self.equity_var.set(f"Equity:  {eq:.2f}")
        else:
            self.equity_var.set("Equity:  N/A")

        if daily_pl is not None and daily_pl_pct is not None:
            sign = "+" if daily_pl >= 0 else ""
            self.daily_pl_var.set(f"Daily P/L: {sign}{daily_pl:.2f} ({daily_pl_pct:.2f}%)")
        else:
            self.daily_pl_var.set("Daily P/L: N/A")

        # Open positions from MT5
        for row in self.pos_tree.get_children():
            self.pos_tree.delete(row)

        positions = self.get_open_positions()
        for p in positions:
            self.pos_tree.insert(
                "",
                "end",
                values=(
                    p["symbol"],
                    p["direction"],
                    f"{p['volume']:.2f}",
                    f"{p['price_open']:.5f}",
                    f"{p['sl']:.5f}" if p["sl"] > 0 else "-",
                    f"{p['tp']:.5f}" if p["tp"] > 0 else "-",
                    f"{p['profit']:.2f}",
                )
            )

        # Prices list
        self.price_list.delete(0, tk.END)
        syms = self.load_symbols_from_yaml()
        prices = self.get_symbol_prices(syms)
        for p in prices:
            if p["spread"] is not None:
                line = f"{p['symbol']:7}  {p['bid']:.5f} / {p['ask']:.5f}  (spr: {p['spread']:.1f})"
            else:
                line = f"{p['symbol']:7}  {p['bid']:.5f} / {p['ask']:.5f}"
            self.price_list.insert(tk.END, line)

    def _start_refresh_loop(self):
        def loop():
            while True:
                try:
                    self._refresh()
                except Exception as e:
                    print(f"[GUI] Refresh error: {e}")
                time.sleep(2)

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    # =====================================================
    # STATUS BAR (Reads service_state.json every 2 seconds)
    # =====================================================
    def init_status_bar(self):
        self.service_status_label = ttk.Label(
            self.window,
            text="Checking system status...",
            background="#222222",
            foreground="#ffffff",
            font=("Segoe UI", 10)
        )
        self.service_status_label.pack(fill="x", pady=4)
        self.update_status_bar()

    def update_status_bar(self):
        try:
            if os.path.exists(SERVICE_STATE_FILE):
                with open(SERVICE_STATE_FILE, "r") as f:
                    state = json.load(f)
                    status = state.get("state", "unknown")
            else:
                status = "unknown"
        except:
            status = "unknown"

        # -------------------------------
        # Map JSON status → visual output
        # -------------------------------
        if status == "running":
            txt = "🟢 REBEL Engine Running"
            color = "#00cc66"
        elif status == "stopped":
            txt = "🔴 REBEL Stopped"
            color = "#ff4444"
        elif status == "starting":
            txt = "🟡 Starting..."
            color = "#ffcc00"
        elif status == "crashed":
            txt = "⚠ REBEL CRASHED"
            color = "#ff8800"
        else:
            txt = "⚪ Unknown Status"
            color = "#cccccc"

        self.service_status_label.config(text=txt, foreground=color)
        # Refresh every 2 seconds
        self.service_status_label.after(2000, self.update_status_bar)

    # -----------------------------
    # CLEANUP
    # -----------------------------
    def on_close(self):
        if self.is_bot_running():
            if not messagebox.askyesno("Exit", "Bot is still running. Stop it and exit?"):
                return
            try:
                self.bot_process.terminate()
            except Exception:
                pass
        self._shutdown_mt5()
        self.window.destroy()


if __name__ == "__main__":
    RebelControlCenter()

