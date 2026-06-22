"""CLI for embedding drift detection.

Usage:
    embedding-drift check --baseline day1.npy --current day2.npy
    embedding-drift check --baseline day1.npy --current day2.npy --threshold cosine_drift=0.5
    embedding-drift plot --store .embedding_drift --output drift.png
    embedding-drift calibrate --baseline day1.npy
"""

import argparse
import sys

import numpy as np

from embedding_drift.core.drift import DriftDetector
from embedding_drift.core.snapshot import SnapshotStore
from embedding_drift.viz.plots import plot_2d_drift


def cmd_check(args):
    ref = np.load(args.baseline)
    cur = np.load(args.current)

    thresholds = None
    if args.threshold:
        thresholds = {}
        for t in args.threshold:
            key, val = t.split("=")
            thresholds[key] = float(val)

    detector = DriftDetector(thresholds=thresholds)

    if args.calibrate:
        print("Auto-calibrating thresholds from baseline...")
        calibrated = detector.calibrate(ref)
        print(f"  Calibrated thresholds: {calibrated}")

    result = detector.check_drift(ref, cur)

    print(f"\nMetrics:")
    for k, v in result["metrics"].items():
        print(f"  {k:<16}: {v:.6f}")

    print(f"\nDrift detected: {'YES' if result['drifted'] else 'NO'}")

    if result["alerts"]:
        print(f"\nAlerts:")
        for a in result["alerts"]:
            print(f"  {a['metric']} = {a['value']:.4f} (threshold: {a['threshold']:.4f})")

    sys.exit(1 if result["drifted"] else 0)


def cmd_plot(args):
    store = SnapshotStore(args.store)
    plot_2d_drift(store, save_path=args.output)


def cmd_calibrate(args):
    ref = np.load(args.baseline)
    detector = DriftDetector()
    thresholds = detector.calibrate(ref, n_splits=args.splits, multiplier=args.multiplier)
    print("Calibrated thresholds:")
    for k, v in thresholds.items():
        print(f"  {k:<16}: {v:.6f}")


def main():
    parser = argparse.ArgumentParser(prog="embedding-drift", description="Embedding drift detection CLI")
    sub = parser.add_subparsers(dest="command")

    # check
    p_check = sub.add_parser("check", help="Check drift between two embedding files")
    p_check.add_argument("--baseline", required=True, help="Path to baseline .npy file")
    p_check.add_argument("--current", required=True, help="Path to current .npy file")
    p_check.add_argument("--threshold", action="append", help="Override threshold: metric=value")
    p_check.add_argument("--calibrate", action="store_true", help="Auto-calibrate thresholds from baseline")

    # plot
    p_plot = sub.add_parser("plot", help="Generate 2D drift visualization")
    p_plot.add_argument("--store", default=".embedding_drift", help="Snapshot store directory")
    p_plot.add_argument("--output", default="drift.png", help="Output image path")

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Auto-calibrate thresholds from reference data")
    p_cal.add_argument("--baseline", required=True, help="Path to reference .npy file")
    p_cal.add_argument("--splits", type=int, default=5, help="Number of folds")
    p_cal.add_argument("--multiplier", type=float, default=2.0, help="Threshold multiplier")

    args = parser.parse_args()

    if args.command == "check":
        cmd_check(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
