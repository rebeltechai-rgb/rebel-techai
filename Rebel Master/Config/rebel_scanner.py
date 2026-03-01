import MetaTrader5 as mt5
import yaml
import os

SYMBOL_LIST_PATH = r"C:\Rebel Technologies\Rebel Master\Config\symbol_lists.yaml"

class RebelScanner:
    def __init__(self):
        self.symbols = self.load_symbols()

    def load_symbols(self):
        if not os.path.exists(SYMBOL_LIST_PATH):
            raise FileNotFoundError("Symbol list is missing.")

        with open(SYMBOL_LIST_PATH, "r") as f:
            data = yaml.safe_load(f)

        groups = data.get("groups", {})
        symbols = []

        for group_name, group_symbols in groups.items():
            symbols.extend(group_symbols)

        print(f"Loaded {len(symbols)} symbols.")
        return symbols

    def test_scanner(self):
        print("SCANNER TEST START")
        for sym in self.symbols:
            print(f"Scanning: {sym}")
        print("Scanner test complete.")
