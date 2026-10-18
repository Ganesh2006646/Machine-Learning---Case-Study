"""Windows live scan: netsh -> fingerprint dict. Linux/Cloud returns []."""
import re
import subprocess


def pct_to_dbm(p):
    try:
        p = max(0.0, min(100.0, float(p)))
        return int(p / 2.0 - 100.0)
    except Exception:
        return -105


def scan_windows(timeout=12):
    try:
        r = subprocess.run(["netsh", "wlan", "show", "networks", "mode=bssid"],
                           capture_output=True, encoding="utf-8",
                           errors="ignore", timeout=timeout)
        out = r.stdout or ""
    except Exception:
        return []
    aps, ssid, bssid = [], "Unknown", None
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^SSID\s+\d+\s*:\s*(.*)", s)
        if m:
            ssid = m.group(1).strip() or "Hidden"
            bssid = None
            continue
        m = re.match(r"^BSSID\s+\d+\s*:\s*([0-9a-fA-F:]{17})", s)
        if m:
            bssid = m.group(1).lower()
            continue
        m = re.match(r"^Signal\s*:\s*(\d+)\s*%", s)
        if m and bssid:
            aps.append({"ssid": ssid, "bssid": bssid,
                        "rssi_dbm": pct_to_dbm(m.group(1))})
            bssid = None
    best = {}
    for a in aps:
        if a["bssid"] not in best or a["rssi_dbm"] > best[a["bssid"]]["rssi_dbm"]:
            best[a["bssid"]] = a
    return list(best.values())


def to_wap_vector(aps, bssid_to_wap, feature_order, fill=-105):
    """Map scanned BSSIDs onto campus WAP feature order."""
    vals = {w: fill for w in feature_order}
    for a in aps:
        w = (bssid_to_wap or {}).get(str(a.get("bssid", "")).lower())
        if w in vals:
            vals[w] = max(vals[w], int(a.get("rssi_dbm", fill)))
    return [vals[w] for w in feature_order]
