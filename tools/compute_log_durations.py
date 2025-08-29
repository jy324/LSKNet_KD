#!/usr/bin/env python3
"""
Compute training durations from log files in given directories.

Usage (from repo root):
  python3 tools/compute_log_durations.py work_dirs/lsk_s_fpn_1x_dota_le90 \
      work_dirs/lsk_s_fpn_1x_dota_le90_pruned work_dirs/lsk_t_fpn_1x_dota_le90

Output:
 - For each folder: list of .log files, per-file start/end timestamps (based on epoch markers if present),
   stitched total span (earliest start epoch timestamp -> latest end epoch timestamp),
   and per-file span summaries.
Notes / assumptions:
 - Recognizes timestamps like '2025-08-25 23:19:01,234', '2025-08-25 23:19:01', '20250825_231901', '2025/08/25 23:19:01'
 - Recognizes epoch patterns like 'Epoch [1/12]', 'epoch: 1', 'Epoch 1' (case-insensitive).
 - If no epoch markers are found, falls back to earliest/latest timestamps in logs.
"""
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta

TS_PATTERNS = [
    re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)"),
    re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"),
    re.compile(r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"),
    re.compile(r"(\d{8}_\d{6})"),
]

EPOCH_PATTERNS = [
    re.compile(r"Epoch\s*\[\s*(\d+)[^\]]*\]", re.IGNORECASE),
    re.compile(r"\bEpoch\b[:\s/]*(\d+)", re.IGNORECASE),
    re.compile(r"epoch[:\s]+(\d+)", re.IGNORECASE),
]

def parse_ts(s: str):
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None

def find_first_ts(line: str):
    for pat in TS_PATTERNS:
        m = pat.search(line)
        if m:
            ts = parse_ts(m.group(1))
            if ts:
                return ts
    return None

def find_epoch(line: str):
    for pat in EPOCH_PATTERNS:
        m = pat.search(line)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def process_log_file(path: Path):
    epoch_timestamps = {}  # epoch -> list[timestamp]
    all_timestamps = []
    try:
        with path.open("r", errors="replace") as f:
            for line in f:
                ts = find_first_ts(line)
                ep = find_epoch(line)
                if ts:
                    all_timestamps.append(ts)
                if ep is not None and ts is not None:
                    epoch_timestamps.setdefault(ep, []).append(ts)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
    return epoch_timestamps, all_timestamps

def compute_folder(folder: Path):
    files = sorted(folder.glob("*.log"))
    if not files:
        return None
    folder_epoch_map = {}
    folder_all_ts = []
    per_file_stats = []

    for f in files:
        ep_map, all_ts = process_log_file(f)
        folder_all_ts.extend(all_ts)
        # record per-file stats
        if ep_map:
            file_epochs = sorted(ep_map.keys())
            start_ep = file_epochs[0]
            end_ep = file_epochs[-1]
            start_ts = min(ep_map[start_ep])
            end_ts = max(ep_map[end_ep])
            per_file_stats.append((f.name, start_ep, end_ep, start_ts, end_ts, end_ts - start_ts))
            for ep, ts_list in ep_map.items():
                folder_epoch_map.setdefault(ep, []).extend(ts_list)
        else:
            if all_ts:
                per_file_stats.append((f.name, None, None, min(all_ts), max(all_ts), max(all_ts) - min(all_ts)))
            else:
                per_file_stats.append((f.name, None, None, None, None, None))

    result = {
        "files": [p.name for p in files],
        "per_file_stats": per_file_stats,
    }

    if folder_epoch_map:
        epochs = sorted(folder_epoch_map.keys())
        min_ep = epochs[0]
        max_ep = epochs[-1]
        start_ts = min(folder_epoch_map[min_ep])
        end_ts = max(folder_epoch_map[max_ep])
        stitched_total = end_ts - start_ts
        # sum per-file spans where available
        active_sum = sum((s[5] for s in per_file_stats if s[5] is not None), timedelta(0))
        result.update({
            "has_epochs": True,
            "min_epoch": int(min_ep),
            "max_epoch": int(max_ep),
            "start_ts": start_ts.isoformat(sep=' '),
            "end_ts": end_ts.isoformat(sep=' '),
            "stitched_total": str(stitched_total),
            "active_sum": str(active_sum),
        })
    else:
        if folder_all_ts:
            start_ts = min(folder_all_ts)
            end_ts = max(folder_all_ts)
            result.update({
                "has_epochs": False,
                "start_ts": start_ts.isoformat(sep=' '),
                "end_ts": end_ts.isoformat(sep=' '),
                "stitched_total": str(end_ts - start_ts),
            })
        else:
            result.update({"has_epochs": False, "start_ts": None, "end_ts": None, "stitched_total": None})

    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 tools/compute_log_durations.py <log_folder1> [<log_folder2> ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        folder = Path(arg)
        print("\nFolder:", folder)
        if not folder.exists():
            print("  Path not found. Provide a relative path from repo root or absolute path.")
            continue
        res = compute_folder(folder)
        if res is None:
            print("  No .log files found")
            continue
        print("  Log files:", res["files"])
        if res.get("has_epochs"):
            print(f"  Epochs: {res['min_epoch']} -> {res['max_epoch']}")
            print(f"  Start ts: {res['start_ts']}")
            print(f"  End ts:   {res['end_ts']}")
            print(f"  Stitched total (end - start): {res['stitched_total']}")
            print(f"  Active sum (per-file epoch spans summed): {res['active_sum']}")
        else:
            print("  No epoch markers found.")
            print(f"  Earliest ts: {res['start_ts']}")
            print(f"  Latest ts:   {res['end_ts']}")
            print(f"  Total span:  {res['stitched_total']}")
        print("  Per-file spans (name, start_ep, end_ep, start_ts, end_ts, span)")
        for p in res.get("per_file_stats", []):
            print("   ", p)
    print("\nDone.")