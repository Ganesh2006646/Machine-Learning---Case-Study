"""
Amrita Campus Wi-Fi Fingerprint Collector (Final)
===================================================
Collects Wi-Fi RSSI fingerprints with AUTO GPS coordinates.

GPS is fetched automatically via Windows Location API.
If auto-GPS fails, it falls back to manual input.

The trained model will predict:
  LATITUDE, LONGITUDE, FLOOR, BLOCK, ROOM

Usage:
    python scripts/campus_collector.py
"""

import subprocess
import re
import time
import datetime
import os
import csv
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "campus")
RAW_LOG_FILE = os.path.join(DATA_DIR, "raw_scans.csv")
GPS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "get_gps.ps1")

CSV_HEADER = [
    "timestamp", "scan_id", "block", "floor", "room",
    "latitude", "longitude", "ssid", "bssid", "rssi_dbm"
]


# =====================================================================
#  GPS Auto-Detection (Windows Location API)
# =====================================================================

def get_gps():
    """
    Get GPS coordinates using Windows Location API via PowerShell.
    Returns (latitude, longitude) or (None, None) if unavailable.
    """
    try:
        r = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", GPS_SCRIPT],
            capture_output=True, encoding="utf-8", errors="ignore", timeout=15
        )
        output = r.stdout.strip()
        if output and output != "UNKNOWN":
            parts = output.split(",")
            lat = float(parts[0])
            lon = float(parts[1])
            acc = float(parts[2]) if len(parts) > 2 else 0
            return lat, lon, acc
    except Exception:
        pass
    return None, None, None


# =====================================================================
#  Wi-Fi Scanner
# =====================================================================

def percent_to_dbm(pct):
    try:
        p = max(0.0, min(100.0, float(pct)))
        return int(p / 2.0 - 100.0)
    except (ValueError, TypeError):
        return -105


def scan_wifi():
    try:
        r = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True, encoding="utf-8", errors="ignore", timeout=10
        )
        output = r.stdout
    except subprocess.TimeoutExpired:
        print("    [!] Scan timed out")
        return []
    except Exception as e:
        print("    [!] Scan error: {}".format(e))
        return []

    if not output or not output.strip():
        return []

    aps = []
    ssid = "Unknown"
    bssid = None

    for line in output.splitlines():
        s = line.strip()
        if not s:
            continue

        m = re.match(r"^SSID\s+\d+\s*:\s*(.*)", s)
        if m:
            ssid = m.group(1).strip() or "Hidden"
            bssid = None
            continue

        m = re.match(r"^BSSID\s+\d+\s*:\s*([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})", s)
        if m:
            bssid = m.group(1).lower()
            continue

        m = re.match(r"^Signal\s*:\s*(\d+)\s*%", s)
        if m and bssid:
            aps.append({
                "ssid": ssid,
                "bssid": bssid,
                "rssi_dbm": percent_to_dbm(int(m.group(1)))
            })
            bssid = None

    return aps


# =====================================================================
#  Helpers
# =====================================================================

def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val):
    if val is None:
        return None
    val = str(val).strip()
    if val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def init_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_LOG_FILE):
        with open(RAW_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print("  Created: {}".format(RAW_LOG_FILE))


def append_rows(rows):
    with open(RAW_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        for row in rows:
            w.writerow(row)


def count_rows():
    try:
        with open(RAW_LOG_FILE, "r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except FileNotFoundError:
        return 0


def quick_test():
    print("\n  [TEST] Quick Wi-Fi scan...")
    aps = scan_wifi()
    if not aps:
        print("  [FAIL] 0 APs. Is Wi-Fi ON?")
        return False
    print("  [OK] {} access points detected:\n".format(len(aps)))
    print("    {:<28s} {:<18s} {:>5s}".format("SSID", "BSSID", "dBm"))
    print("    {} {} {}".format("-" * 28, "-" * 18, "-" * 5))
    for a in aps:
        print("    {:<28s} {:<18s} {:>4d}".format(a["ssid"], a["bssid"], a["rssi_dbm"]))
    print()
    return True


def gps_test():
    print("  [TEST] Auto GPS detection...")
    lat, lon, acc = get_gps()
    if lat is not None:
        print("  [OK] GPS: {}, {} (accuracy: {}m)".format(lat, lon, int(acc)))
        return True
    else:
        print("  [!] Auto GPS unavailable. You'll need to enter coordinates manually.")
        print("      To fix: Windows Settings > Privacy > Location > Turn ON")
        return False


# =====================================================================
#  Main
# =====================================================================

def main():
    init_csv()

    print("=" * 65)
    print("  AMRITA CAMPUS WI-FI FINGERPRINT COLLECTOR")
    print("=" * 65)
    print("  Output: {}".format(RAW_LOG_FILE))
    print()

    # Pre-flight checks
    if not quick_test():
        print("  Enable Wi-Fi and restart.")
        return

    gps_available = gps_test()
    print()

    # Handle existing data
    n = count_rows()
    if n > 0:
        print("  [INFO] {} existing rows in file.".format(n))
        ch = input("  (A)ppend or (S)tart fresh? [A]: ").strip().upper()
        if ch == "S":
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = os.path.join(DATA_DIR, "archive_{}.csv".format(ts))
            shutil.copy(RAW_LOG_FILE, bak)
            print("  [OK] Backed up to: {}".format(bak))
            with open(RAW_LOG_FILE, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADER)
            print("  [OK] Started fresh.\n")

    # Session setup
    print("-" * 65)
    block = input("  BLOCK name (e.g. D, A, B): ").strip().upper()
    if not block:
        block = "D"

    bi = input("  Scans per spot? [15]: ").strip()
    bursts = safe_int(bi, 15)
    bursts = max(1, min(100, bursts))

    print("\n  Session: Block={}, Bursts={}".format(block, bursts))
    if gps_available:
        print("  GPS: AUTO (coordinates fetched automatically at each spot)")
    else:
        print("  GPS: MANUAL (you will type lat/lon at each spot)")
    print("  Type 'q' at floor prompt to finish.\n")

    # Collection loop
    spots = 0

    while True:
        print("-" * 65)
        fi = input("  FLOOR (e.g. 0, 1, 2, 3) [q=quit]: ").strip()
        if fi.lower() == "q":
            break

        floor = safe_int(fi, -1)
        if floor < 0:
            print("  [!] Enter a valid floor number.")
            continue

        room = input("  ROOM (e.g. 301, lab1, corridor): ").strip()
        if not room:
            room = "unknown"
        room = room.replace(",", "_").replace('"', "").strip()

        # GPS coordinates
        lat = None
        lon = None

        if gps_available:
            print("  Fetching GPS...")
            lat, lon, acc = get_gps()
            if lat is not None:
                print("  [OK] Auto GPS: {}, {} (accuracy: {}m)".format(
                    round(lat, 7), round(lon, 7), int(acc)))
                override = input("  Press Enter to accept, or type 'lat,lon' to override: ").strip()
                if override and "," in override:
                    parts = override.split(",")
                    lat = safe_float(parts[0].strip())
                    lon = safe_float(parts[1].strip())
                    if lat is None or lon is None:
                        print("  [!] Invalid override. Using auto GPS.")
                        lat, lon, _ = get_gps()
            else:
                print("  [!] Auto GPS failed this time. Enter manually:")
                gps_available = False  # Switch to manual for rest of session

        if lat is None:
            lat_in = input("  LATITUDE  (e.g. 10.90366): ").strip()
            lat = safe_float(lat_in)
            if lat is None:
                print("  [!] Latitude is required.")
                continue

            lon_in = input("  LONGITUDE (e.g. 76.90026): ").strip()
            lon = safe_float(lon_in)
            if lon is None:
                print("  [!] Longitude is required.")
                continue

        room_label = "{}-{}".format(block, room)
        print("\n  >> {}  Floor={}  GPS=({}, {})".format(
            room_label, floor, round(lat, 7), round(lon, 7)))
        print("  >> Recording {} bursts...".format(bursts))

        records = []
        empty = 0

        for i in range(1, bursts + 1):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sid = "{}_F{}_{}_{}_{:03d}".format(
                block, floor, room,
                now.replace(" ", "T").replace(":", ""), i
            )
            aps = scan_wifi()

            if not aps:
                empty += 1
                print("    [{:02d}/{:02d}] 0 APs".format(i, bursts))
            else:
                for a in aps:
                    records.append({
                        "timestamp": now,
                        "scan_id": sid,
                        "block": block,
                        "floor": floor,
                        "room": room_label,
                        "latitude": round(lat, 7),
                        "longitude": round(lon, 7),
                        "ssid": a["ssid"],
                        "bssid": a["bssid"],
                        "rssi_dbm": a["rssi_dbm"]
                    })
                print("    [{:02d}/{:02d}] {} APs".format(i, bursts, len(aps)))

            if i < bursts:
                time.sleep(1.5)

        if records:
            append_rows(records)
            uniqs = len(set(r["bssid"] for r in records))
            spots += 1
            print("\n  [OK] Saved {} rows ({} unique APs)".format(len(records), uniqs))
            if empty:
                print("  [!] {}/{} empty scans".format(empty, bursts))
            print("  Spots done: {}\n".format(spots))
        else:
            print("\n  [FAIL] No APs detected.\n")

    # Summary
    total = count_rows()
    print()
    print("=" * 65)
    print("  DONE! Spots: {}  |  Total rows: {}".format(spots, total))
    print("  File: {}".format(RAW_LOG_FILE))
    print("  Next: python scripts/build_campus_dataset.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
