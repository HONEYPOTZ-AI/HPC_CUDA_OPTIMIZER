#!/usr/bin/env python3
"""
check_regression.py — Parse benchmark output and compare against a stored baseline.

Usage:
    python3 check_regression.py \
        --current  benchmark_results.txt \
        --baseline benchmark_baseline.txt \
        --threshold 10          # fail if any metric regressed > 10%

Exit codes:
    0  All metrics within threshold
    1  One or more metrics regressed beyond threshold
    2  Parse error (no numeric data found in files)
"""

import argparse
import re
import sys
from typing import Dict, Optional

# Regex to extract "label : value unit" or "label = value" patterns
METRIC_RE = re.compile(
    r"(?P<label>[A-Za-z][\w\s\-/]+?)"   # metric name
    r"\s*[:\|=]\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)"    # numeric value
    r"\s*(?P<unit>[A-Za-z/]*)",           # optional unit
    re.MULTILINE,
)

# Keywords that identify throughput / timing metrics we care about
METRIC_KEYWORDS = [
    "throughput", "bandwidth", "step", "ms/step", "GB/s", "TFlops",
    "iter", "samples/s", "tokens/s", "mean", "p95", "min", "max",
]


def parse_metrics(path: str) -> Dict[str, float]:
    """Extract numeric metrics from a benchmark log file."""
    metrics: Dict[str, float] = {}
    try:
        with open(path, "r", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(2)

    for m in METRIC_RE.finditer(content):
        label = m.group("label").strip().lower()
        # Only keep metrics related to performance
        if any(kw in label for kw in METRIC_KEYWORDS):
            key = re.sub(r"\s+", "_", label)
            try:
                metrics[key] = float(m.group("value"))
            except ValueError:
                pass

    return metrics


def compare(current: Dict[str, float], baseline: Dict[str, float],
            threshold_pct: float) -> bool:
    """
    Compare current metrics against baseline.
    Returns True if all metrics are within threshold, False if any regressed.
    """
    regressions = []
    improvements = []
    unchanged = []

    all_keys = set(current) | set(baseline)

    for key in sorted(all_keys):
        cur_val = current.get(key)
        base_val = baseline.get(key)

        if cur_val is None:
            print(f"  ⚠  MISSING in current  : {key} (baseline={base_val:.2f})")
            continue
        if base_val is None:
            print(f"  ℹ  NEW metric           : {key} = {cur_val:.2f}")
            continue

        if base_val == 0:
            pct = 0.0
        else:
            pct = (cur_val - base_val) / abs(base_val) * 100.0

        # For throughput metrics: higher is better → negative pct is regression
        # For timing metrics (ms, latency): lower is better → positive pct is regression
        is_timing = any(t in key for t in ["ms", "latency", "time", "step"])
        regressed = (pct < -threshold_pct) if not is_timing else (pct > threshold_pct)
        improved  = (pct > threshold_pct) if not is_timing else (pct < -threshold_pct)

        direction = "↓" if pct < 0 else "↑"
        marker = "✗ REGRESSION" if regressed else ("✓ improved  " if improved else "  stable    ")

        print(f"  {marker}  {key:45s}  baseline={base_val:8.2f}  current={cur_val:8.2f}  {direction}{abs(pct):.1f}%")

        if regressed:
            regressions.append((key, base_val, cur_val, pct))
        elif improved:
            improvements.append(key)
        else:
            unchanged.append(key)

    print()
    print(f"  Summary: {len(regressions)} regressions | "
          f"{len(improvements)} improvements | {len(unchanged)} unchanged")

    if regressions:
        print()
        print(f"  ✗ FAILED — {len(regressions)} metric(s) exceeded {threshold_pct}% regression threshold:")
        for key, base, cur, pct in regressions:
            print(f"      {key}: {base:.2f} → {cur:.2f}  ({pct:+.1f}%)")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Performance regression checker")
    parser.add_argument("--current",   required=True, help="Current benchmark log")
    parser.add_argument("--baseline",  required=True, help="Baseline benchmark log")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Regression threshold in percent (default: 10)")
    args = parser.parse_args()

    print(f"Performance Regression Check (threshold: {args.threshold}%)")
    print(f"  Current  : {args.current}")
    print(f"  Baseline : {args.baseline}")
    print()

    current  = parse_metrics(args.current)
    baseline = parse_metrics(args.baseline)

    if not current:
        print("ERROR: No metrics found in current results file.", file=sys.stderr)
        sys.exit(2)
    if not baseline:
        print("WARNING: No metrics found in baseline file — skipping regression check.")
        sys.exit(0)

    print(f"  Parsed {len(current)} metrics from current, {len(baseline)} from baseline")
    print()

    passed = compare(current, baseline, args.threshold)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
