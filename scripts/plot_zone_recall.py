#!/usr/bin/env python3
"""Per-zone SWIFI recall vs matrix size.

Reads abft_zone_recall.csv (the metrics CSV produced by repeated
`--inject swifi --swifi-zone Z` runs) and plots recall as a function of
matrix size, one line per IEEE-754 bit zone.  This is the figure that
explains the false negatives: sig_low stays near zero, everything else
near one.
"""
import argparse, csv, sys
from collections import defaultdict

ZONE_ORDER = ["sign", "exponent", "sig_high", "sig_low", "any"]
ZONE_STYLE = {
    "sign":     ("tab:red",    "o", "sign (bit 31)"),
    "exponent": ("tab:orange", "s", "exponent (23-30)"),
    "sig_high": ("tab:blue",   "D", "significand high (13-22)"),
    "sig_low":  ("tab:gray",   "X", "significand low (0-12)"),
    "any":      ("tab:green",  "^", "any (0-31)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed", file=sys.stderr)
        sys.exit(2)

    # zone -> list of (size, recall)
    series = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            z = row.get("swifi_zone", "any")
            try:
                size = int(row["M"])
                recall = float(row["recall"])
            except (KeyError, ValueError):
                continue
            series[z].append((size, recall))

    if not series:
        print(f"ERROR: no rows in {args.csv}", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for z in ZONE_ORDER:
        if z not in series:
            continue
        pts = sorted(series[z])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color, marker, label = ZONE_STYLE.get(z, ("black", ".", z))
        ax.plot(xs, ys, "-", color=color, marker=marker, markersize=8,
                linewidth=2.4, label=label)

    ax.set_xlabel("Matrix size (M = N = K)")
    ax.set_ylabel("Recall = TP / (TP + FN)")
    ax.set_title("SWIFI recall per IEEE-754 bit zone")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=9, title="bit zone")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
