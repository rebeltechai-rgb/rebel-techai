import pandas as pd
import os

# === PATHS ===
BASE_PATH = r"C:\Rebel Technologies\Rebel Master\ML"
INPUT_FILE = os.path.join(BASE_PATH, "features.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "features_clean.csv")

print("=== CLEANING FEATURES.CSV ===")

# Load dataset
df = pd.read_csv(INPUT_FILE)

print(f"[INFO] Loaded {len(df)} rows")

# Fill missing values
df["macd_hist"] = df["macd_hist"].fillna(0)
df["spread_ratio"] = df["spread_ratio"].fillna(0)
df["reason"] = df["reason"].fillna("none")

# Ensure floats are valid numeric
float_cols = ["ema_fast", "ema_slow", "rsi", "atr", "adx",
              "macd_hist", "volatility_regime", "spread_ratio",
              "raw_signal", "signal_score"]

for col in float_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Save clean version
df.to_csv(OUTPUT_FILE, index=False)

print("=== CLEANING COMPLETE ===")
print(f"[INFO] Saved cleaned dataset → {OUTPUT_FILE}")
print("[INFO] Missing values filled and numeric columns normalized")

