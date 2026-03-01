import pandas as pd
import os

# ------------------------------------
# SYSTEM CONFIG
# ------------------------------------
BASE_PATH = r"C:\Rebel Technologies\Rebel Master\ML"
INPUT_FILE = os.path.join(BASE_PATH, "features_clean.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "features_rebuilt.csv")

print("=== REBUILDING FEATURES ===")

df = pd.read_csv(INPUT_FILE)
print(f"[INFO] Loaded {len(df)} rows")

# ------------------------------------
# REBUILD MACD HISTOGRAM
# ------------------------------------
def compute_macd_hist(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

# If close prices exist, rebuild; if not, fill with 0
if "close" in df.columns:
    try:
        df["macd_hist"] = compute_macd_hist(df["close"])
        df["macd_hist"] = df["macd_hist"].fillna(0)
        print("[OK] Rebuilt MACD histogram")
    except:
        df["macd_hist"] = 0
else:
    df["macd_hist"] = 0


# ------------------------------------
# REBUILD SPREAD RATIO
# ------------------------------------
def compute_spread_ratio(spread, atr):
    if atr <= 0:
        return 0
    return spread / atr

if "atr" in df.columns and "spread_ratio" in df.columns:
    df["spread_ratio"] = df.apply(
        lambda row: compute_spread_ratio(row.get("spread_ratio", 0), row["atr"]),
        axis=1
    )
    print("[OK] Rebuilt Spread Ratio")

# ------------------------------------
# REASON REBUILD
# ------------------------------------
def rebuild_reason(row):
    if isinstance(row.get("reason"), str) and row["reason"] != "":
        return row["reason"]

    # fallback rules
    if row.get("signal_score", 0) >= 4:
        return "strong_signal"
    if row.get("trend_bias", 0) != 0:
        return "trend_signal"
    return "none"

df["reason"] = df.apply(rebuild_reason, axis=1)

print("[OK] Rebuilt Reason field")

# ------------------------------------
# SAVE
# ------------------------------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"=== REBUILD COMPLETE → {OUTPUT_FILE} ===")

