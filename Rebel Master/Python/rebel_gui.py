import json
import os
import tkinter as tk
from tkinter import ttk, messagebox

DASHBOARD_PATH = r"C:\Rebel Technologies\Rebel Master\Config\dashboard_config.json"


class RebelDashboard:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("REBEL Master Control Panel")
        self.window.geometry("420x520")
        self.window.configure(bg="#111")

        self.settings = self.load_settings()

        self.build_ui()
        self.window.mainloop()

    # -----------------------------------
    # File Handling
    # -----------------------------------
    def load_settings(self):
        if not os.path.exists(DASHBOARD_PATH):
            return {}

        try:
            with open(DASHBOARD_PATH, "r") as f:
                return json.load(f)
        except:
            return {}

    def save_settings(self):
        try:
            with open(DASHBOARD_PATH, "w") as f:
                json.dump(self.settings, f, indent=4)
            messagebox.showinfo("Saved", "Settings updated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dashboard settings:\n{e}")

    # -----------------------------------
    # UI
    # -----------------------------------
    def build_ui(self):
        title = tk.Label(
            self.window,
            text="REBEL MASTER CONTROL PANEL",
            bg="#111",
            fg="#00FF88",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=15)

        # Auto-trade toggle
        self.auto_var = tk.BooleanVar(value=self.settings.get("auto_trade", False))
        chk = tk.Checkbutton(
            self.window,
            text="Enable Auto-Trading",
            variable=self.auto_var,
            bg="#111",
            fg="white",
            selectcolor="#222",
            font=("Arial", 12),
            command=self.update_auto_trade
        )
        chk.pack(pady=10)

        # Strategy mode selector
        tk.Label(self.window, text="Strategy Mode:", bg="#111", fg="white").pack()
        self.mode_var = tk.StringVar(value=self.settings.get("strategy_mode", "normal"))
        ttk.Combobox(
            self.window,
            textvariable=self.mode_var,
            values=["conservative", "normal", "aggressive"]
        ).pack(pady=5)

        # Min Score
        tk.Label(self.window, text="Minimum Signal Score:", bg="#111", fg="white").pack()
        self.score_entry = tk.Entry(self.window, width=5)
        self.score_entry.insert(0, str(self.settings.get("min_score", 3)))
        self.score_entry.pack(pady=5)

        # Timeframe
        tk.Label(self.window, text="Scanner Timeframe:", bg="#111", fg="white").pack()
        self.tf_var = tk.StringVar(value=self.settings.get("timeframe", "M15"))
        ttk.Combobox(
            self.window,
            textvariable=self.tf_var,
            values=["M1", "M5", "M15", "M30", "H1", "H4"]
        ).pack(pady=5)

        # Add symbol
        tk.Label(self.window, text="Add Symbol:", bg="#111", fg="white").pack(pady=5)
        self.symbol_entry = tk.Entry(self.window, width=15)
        self.symbol_entry.pack()

        tk.Button(
            self.window,
            text="Add Symbol",
            command=self.add_symbol,
            bg="#333",
            fg="white",
            width=20
        ).pack(pady=5)

        # KILL SWITCH
        tk.Button(
            self.window,
            text="⚠ EMERGENCY STOP",
            command=self.kill_switch,
            bg="#880000",
            fg="white",
            width=22,
            font=("Arial", 12, "bold")
        ).pack(pady=12)

        # Save
        tk.Button(
            self.window,
            text="Save Settings",
            command=self.save_all,
            bg="#0077FF",
            fg="white",
            width=20
        ).pack(pady=15)

    # -----------------------------------
    # Actions
    # -----------------------------------
    def update_auto_trade(self):
        self.settings["auto_trade"] = self.auto_var.get()

    def add_symbol(self):
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            return

        if "add_symbols" not in self.settings:
            self.settings["add_symbols"] = []

        self.settings["add_symbols"].append(symbol)
        messagebox.showinfo("Symbol Added", f"{symbol} added to scan list!")

        self.symbol_entry.delete(0, tk.END)

    def kill_switch(self):
        self.settings["kill_switch"] = True
        self.save_settings()
        messagebox.showwarning("STOP", "Emergency stop activated! All trading halted.")

    def save_all(self):
        self.settings["strategy_mode"] = self.mode_var.get()
        self.settings["min_score"] = int(self.score_entry.get())
        self.settings["timeframe"] = self.tf_var.get()
        self.settings["auto_trade"] = self.auto_var.get()
        self.save_settings()


if __name__ == "__main__":
    RebelDashboard()

