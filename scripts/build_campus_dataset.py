"""
Amrita Campus Dataset Builder (v2)
==================================
Transforms raw long-format Wi-Fi scans (from campus_collector.py)
into a clean, wide-format ML-ready dataset matching UJIndoorLoc structure.

Supports:
  - Multi-block campus layout (BLOCK -> BUILDINGID)
  - Auto-coordinates (X, Y)
  - Full WAP feature matrix with -105 dBm sentinel fill
  - WAP dictionary mapping (hardware BSSID -> WAP name + SSID)

Usage:
    python scripts/build_campus_dataset.py

Input:
    data/campus/raw_scans.csv

Output:
    data/campus/trainingData_campus.csv
    data/campus/wap_dictionary.json
"""

import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "campus")
RAW_LOG_FILE = os.path.join(DATA_DIR, "raw_scans.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "trainingData_campus.csv")
WAP_MAP_JSON = os.path.join(DATA_DIR, "wap_dictionary.json")

SENTINEL_DBM = -105  # Standard value for "not detected"


def process_campus_data():
    # --- 1. Load and validate raw data ---
    if not os.path.exists(RAW_LOG_FILE):
        print(f"Error: Raw log file not found at {RAW_LOG_FILE}")
        print("Run `python scripts/campus_collector.py` first to collect scans.")
        return None

    df_raw = pd.read_csv(RAW_LOG_FILE)

    if df_raw.empty:
        print("Raw log file is empty. Please collect some scans first.")
        return None

    # Drop rows with missing critical fields
    before_count = len(df_raw)
    df_raw = df_raw.dropna(subset=["scan_id", "bssid", "rssi_dbm"])
    dropped = before_count - len(df_raw)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing scan_id/bssid/rssi_dbm.")

    # Normalize BSSIDs to lowercase
    df_raw["bssid"] = df_raw["bssid"].astype(str).str.lower().str.strip()

    # Backwards compatibility: if 'block' missing, infer from room
    if "block" not in df_raw.columns:
        df_raw["block"] = df_raw["room"].astype(str).str[0].str.upper()
    else:
        df_raw["block"] = df_raw["block"].fillna("").astype(str).str.strip().str.upper()
        # Fallback for empty blocks
        df_raw.loc[df_raw["block"] == "", "block"] = (
            df_raw.loc[df_raw["block"] == "", "room"].astype(str).str[0].str.upper()
        )

    # Ensure floor is integer
    df_raw["floor"] = pd.to_numeric(df_raw["floor"], errors="coerce").fillna(0).astype(int)

    # Ensure rssi_dbm is integer
    df_raw["rssi_dbm"] = pd.to_numeric(df_raw["rssi_dbm"], errors="coerce").fillna(SENTINEL_DBM).astype(int)

    # Ensure latitude, longitude are float
    for col in ["latitude", "longitude"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0.0)

    print("=" * 65)
    print("  PROCESSING RAW CAMPUS WI-FI SCANS (v2)")
    print("=" * 65)
    print(f"  Total raw observation rows : {len(df_raw)}")
    print(f"  Unique scans (samples)     : {df_raw['scan_id'].nunique()}")

    # --- 2. Build Master WAP Dictionary ---
    unique_bssids = sorted(df_raw["bssid"].unique())
    print(f"  Unique Access Points (BSSIDs): {len(unique_bssids)}")

    wap_mapping = {}    # WAP001 -> {bssid, ssid}
    bssid_to_wap = {}   # bssid -> WAP001

    for idx, bssid in enumerate(unique_bssids, 1):
        wap_name = f"WAP{idx:03d}"
        bssid_to_wap[bssid] = wap_name

        # Find most common SSID for this BSSID
        ssid_series = df_raw.loc[df_raw["bssid"] == bssid, "ssid"].dropna()
        if not ssid_series.empty:
            ssid_mode = ssid_series.mode()
            ssid_label = ssid_mode.iloc[0] if not ssid_mode.empty else "Unknown"
        else:
            ssid_label = "Unknown"

        wap_mapping[wap_name] = {
            "bssid": bssid,
            "ssid": str(ssid_label)
        }

    # Save WAP Dictionary
    with open(WAP_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(wap_mapping, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved WAP Dictionary: {WAP_MAP_JSON}")

    # --- 3. Map BSSIDs to WAP names ---
    df_raw["wap_name"] = df_raw["bssid"].map(bssid_to_wap)

    # --- 4. Pivot to wide format ---
    pivot_df = df_raw.pivot_table(
        index="scan_id",
        columns="wap_name",
        values="rssi_dbm",
        aggfunc="max"
    )

    # Fill undetected APs with sentinel
    pivot_df = pivot_df.fillna(SENTINEL_DBM).astype(int)

    # --- 5. Attach metadata ---
    meta_df = df_raw.groupby("scan_id").agg({
        "block": "first",
        "floor": "first",
        "room": "first",
        "latitude": "first",
        "longitude": "first",
        "timestamp": "first"
    })

    # Add numeric BUILDINGID (like UJIndoorLoc: 0, 1, 2...)
    unique_blocks = sorted(meta_df["block"].unique())
    block_to_id = {b: i for i, b in enumerate(unique_blocks)}
    meta_df["building_id"] = meta_df["block"].map(block_to_id).fillna(0).astype(int)

    # Merge on scan_id index
    final_df = pivot_df.join(meta_df, how="left").reset_index()

    # --- 6. Reorder columns: WAPs first, then metadata ---
    wap_cols = sorted([c for c in final_df.columns if c.startswith("WAP")])
    meta_cols = ["block", "building_id", "floor", "room", "latitude", "longitude", "timestamp", "scan_id"]

    meta_cols = [c for c in meta_cols if c in final_df.columns]
    final_df = final_df[wap_cols + meta_cols]

    # Type enforcement
    final_df["floor"] = final_df["floor"].astype(int)
    final_df["building_id"] = final_df["building_id"].astype(int)

    # --- 7. Save ---
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  [OK] Dataset saved: {OUTPUT_CSV}")
    print(f"    Shape: {final_df.shape[0]} samples x {len(wap_cols)} WAP features")
    print(f"\n  Block (Building) distribution:")
    print(final_df["block"].value_counts().to_string(header=False))
    print(f"\n  Floor distribution:")
    print(final_df["floor"].value_counts().sort_index().to_string(header=False))
    print(f"\n  Room distribution:")
    print(final_df["room"].value_counts().to_string(header=False))
    print("=" * 65)

    return final_df


if __name__ == "__main__":
    process_campus_data()
