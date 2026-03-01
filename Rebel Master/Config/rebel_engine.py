import MetaTrader5 as mt5
import yaml
import os

CONFIG_PATH = r"C:\Rebel Technologies\Rebel Master\Config\master_config.yaml"

class RebelEngine:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    def connect_mt5(self):
        cfg = self.config.get("mt5", {})

        if not mt5.initialize():
            raise RuntimeError("MT5 failed to initialize")

        print("MT5 connected successfully.")
        return True

    def start(self):
        print("REBEL ENGINE - CLEAN START")
        self.connect_mt5()
        print("Engine ready.")
