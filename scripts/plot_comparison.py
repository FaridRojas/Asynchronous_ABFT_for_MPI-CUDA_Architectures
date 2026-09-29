#!/usr/bin/env python3
"""Per-regime comparison plots vs Fault-Tolerant-SGEMM.

Input: ``compare_all.csv`` with one row per (regime, M, K, N, who) and
columns ``gflops`` and ``time_ms``.  ``who`` ranges over:
    ours_baseline    — our cuBLAS, no ABFT
    ours_online      — our ABFT, no fault
    ours_online_inj  — our ABFT, fault (additive)
    theirs_cublas    — their cuBLAS, no ABFT
    theirs_fused     — their fused ABFT, no fault
    theirs_fused_inj — their fused ABFT, fault (additive)

For every regime the plotter emits FOUR PNGs:
    gflops_<regime>.png             throughput (GFLOPS) — all 6 series
    time_<regime>.png               runtime  (ms)      — all 6 series
    overhead_vs_ours_<regime>.png   four ABFT impls' runtime overhead
                                    vs OUR cuBLAS baseline (%)
    overhead_vs_theirs_<regime>.png same four overheads vs THEIR cuBLAS

Backwards-compat: if ``time_ms`` is missing, time is derived from
gflops as t_ms = 2*M*N*K/(gflops*1e9)*1e3.
"""
import argparse, csv, os, sys
from collections import defaultdict

WHO_STYLE = {
    "ours_baseline":    ("tab:green",  "s", "ours / baseline cuBLAS"),
    "ours_online":      ("tab:red",    "D", "ours / online ABFT (no fault)"),
    "ours_online_inj":  ("tab:orange", "D", "ours / online ABFT (fault)"),
    "theirs_cublas":    ("tab:blue",   "o", "theirs / cuBLAS"),
    "theirs_fused":     ("tab:gray",   "^", "theirs / fused ABFT (no fault)"),
    "theirs_fused_inj": ("tab:purple", "^", "theirs / fused ABFT (fault)"),
}
REGIME_TITLE = {
    "small_sq": "Square, small (M=N=K, ~50-1024)",
    "big_sq":   "Square, big (M=N=K, 256-10240)",
    "small_ns": "Non-square, small (M=N, K=256)",
    "big_ns":   "Non-square, big (M=N, K=1024, 256-10240)",
}


def eff_size(M, K, N):
    return round((M * K * N) ** (1.0 / 3.0))


def _ms_from_gflops(M, K, N, gflops):
    if gflops <= 0:
        return 0.0
    return 2.0 * M * N * K / (gflops * 1e9) * 1e3


def _xticks(ax, xticks, xlabels, regime):
    n = len(xticks)
    step = max(1, round(n / 10))
    shown = list(range(0, n, step))
    if (n - 1) not in shown:
        shown.append(n - 1)
    ax.set_xticks([xticks[i] for i in shown])
    ax.set_xticklabels([xlabels[i] for i in shown],
                       rotation=30, ha="right", fontsize=8)
    left = 0 if regime.startswith("big") else min(xticks) * 0.97
    ax.set_xlim(left, max(xticks) * 1.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        print("matplotlib not installed", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.outdir, exist_ok=True)

    # regime -> who -> list of (xval, label, gflops, time_ms, M, K, N)
    data = defaultdict(lambda: defaultdict(list))
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            try:
                M, K, N = int(r["M"]), int(r["K"]), int(r["N"])
                g = float(r["gflops"])
            except (KeyError, ValueError):
                continue
            try:
                t_ms = float(r.get("time_ms") or "")
            except ValueError:
                t_ms = _ms_from_gflops(M, K, N, g)
            if M == K == N:
                x, lbl = M, str(M)
            elif M == N:
                x, lbl = M, f"{M}(K={K})"
            else:
                x, lbl = eff_size(M, K, N), f"{M}x{K}x{N}"
            data[r["regime"]][r["who"]].append(
                (x, lbl, g, t_ms, M, K, N))

    if not data:
        print(f"ERROR: no rows in {args.csv}", file=sys.stderr)
        sys.exit(1)

    for regime, whos in data.items():
        title = REGIME_TITLE.get(regime, regime)
        xaxis_lbl = ("Matrix size  (M=N=K)" if regime.endswith("sq")
                     else "M = N  (K fixed)")

        # ============ GFLOPS figure ============
        fig, ax = plt.subplots(figsize=(9, 5.5))
        xticks, xlabels = None, None
        for who, (color, marker, label) in WHO_STYLE.items():
            if who not in whos:
                continue
            pts = sorted(whos[who])
            xs = [p[0] for p in pts]
            ys = [p[2] for p in pts]
            ax.plot(xs, ys, "-", color=color, marker=marker, markersize=7,
                    linewidth=2.3, label=label)
            if xticks is None or len(xs) > len(xticks):
                xticks = xs
                xlabels = [p[1] for p in pts]
        ax.set_xlabel(xaxis_lbl)
        ax.set_ylabel("GFLOPS/s")
        ax.set_title(f"GFLOPS — {title}")
        if xticks:
            _xticks(ax, xticks, xlabels, regime)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"gflops_{regime}.png"), dpi=150)
        plt.close(fig)

        # ============ Runtime (ms) figure ============
        fig, ax = plt.subplots(figsize=(9, 5.5))
        xticks, xlabels = None, None
        for who, (color, marker, label) in WHO_STYLE.items():
            if who not in whos:
                continue
            pts = sorted(whos[who])
            xs = [p[0] for p in pts]
            ys = [p[3] for p in pts]
            ax.plot(xs, ys, "-", color=color, marker=marker, markersize=7,
                    linewidth=2.3, label=label)
            if xticks is None or len(xs) > len(xticks):
                xticks = xs
                xlabels = [p[1] for p in pts]
        ax.set_xlabel(xaxis_lbl)
        ax.set_ylabel("Runtime (ms)")
        ax.set_title(f"Runtime — {title}")
        if xticks:
            _xticks(ax, xticks, xlabels, regime)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"time_{regime}.png"), dpi=150)
        plt.close(fig)

        # ============ Overhead-vs-baseline figures ============
        # For each of OUR cuBLAS and THEIR cuBLAS, plot the runtime overhead
        # (%) of the four ABFT implementations (ours/theirs × no-fault/fault).
        def time_by_x(who):
            return {p[0]: p[3] for p in whos.get(who, [])}

        abft_curves = [
            ("ours_online",      "tab:red",    "-D",
             "ours / ABFT (no fault)"),
            ("ours_online_inj",  "tab:orange", "--D",
             "ours / ABFT (fault)"),
            ("theirs_fused",     "tab:gray",   "-^",
             "theirs / ABFT (no fault)"),
            ("theirs_fused_inj", "tab:purple", "--^",
             "theirs / ABFT (fault)"),
        ]

        for ref_who, ref_short, tag in (("ours_baseline", "our cuBLAS",
                                         "ours"),
                                        ("theirs_cublas", "their cuBLAS",
                                         "theirs")):
            ref = time_by_x(ref_who)
            if not ref:
                continue
            fig, ax = plt.subplots(figsize=(9, 5.5))
            allx = set()
            for who, color, fmt, label in abft_curves:
                abft = time_by_x(who)
                xs = sorted(set(ref) & set(abft))
                if not xs:
                    continue
                ov = [(abft[x] / ref[x] - 1.0) * 100.0 for x in xs]
                ax.plot(xs, ov, fmt, color=color, linewidth=2.3,
                        markersize=7, label=label)
                allx |= set(xs)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.set_xlabel(xaxis_lbl)
            ax.set_ylabel(f"Runtime overhead vs {ref_short} (%)")
            ax.set_title(f"ABFT overhead vs {ref_short} — {title}")
            allx = sorted(allx)
            if allx:
                left = 0 if regime.startswith("big") else min(allx) * 0.95
                ax.set_xlim(left, max(allx) * 1.05)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir,
                                     f"overhead_vs_{tag}_{regime}.png"),
                        dpi=150)
            plt.close(fig)

        print(f"  {regime}: gflops_{regime}.png  time_{regime}.png  "
              f"overhead_vs_ours_{regime}.png  overhead_vs_theirs_{regime}.png")


if __name__ == "__main__":
    main()
