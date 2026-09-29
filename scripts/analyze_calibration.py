#!/usr/bin/env python3
"""
analyze_calibration.py — derive the detection threshold from calibration data.

WHY THIS EXISTS
---------------
tau used to be "observed max |a-e| x 10".  The 10 was arbitrary, and an
arbitrary constant in the middle of a detection result is not defensible: it
is neither a bound nor a measurement, and it silently costs recall (a larger
tau hides every fault smaller than it).

This script replaces it with a stated criterion.  Each calibration pass runs a
clean protected GEMM and dumps every |actual - expected| checksum reading, so
one pass gives the maximum a single protected GEMM would have to clear to
avoid a false positive.  With R independent passes we get R samples of that
maximum, which is exactly the block-maxima setting of extreme-value theory:
fit a Gumbel distribution to them and tau follows from a false-positive BUDGET
instead of a guess.

    tau(c) = c * max_observed
    P(false positive per protected GEMM) = 1 - exp(-exp(-(tau - mu) / beta))

The reported factor is the smallest integer whose fitted false-positive
probability stays under the budget at EVERY calibrated size.  Integer, because
the extra precision of a fractional factor is not real: mu and beta come from
R = 20 samples.

USAGE
-----
    python3 scripts/analyze_calibration.py --diffs-dir docs/calibration
    python3 scripts/analyze_calibration.py --diffs-dir docs/calibration \
            --emit-table scripts/tau_table.generated.sh \
            --emit-tex   scripts/tau_table.tex
    python3 scripts/analyze_calibration.py --diffs-dir docs/calibration --recall

--recall adds the other side of the trade-off (see recall_model below).
"""

import argparse
import csv
import glob
import math
import os
import re
import sys

import numpy as np

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Reading the dumps
# ---------------------------------------------------------------------------
def load_diffs(diffs_dir):
    """{size: 1-D array of |a-e|} from abft_calib_diffs_<N>.csv."""
    out = {}
    pattern = os.path.join(diffs_dir, "abft_calib_diffs_*.csv")
    for path in glob.glob(pattern):
        m = re.search(r"_(\d+)\.csv$", path)
        if not m:
            continue
        vals = np.loadtxt(path, skiprows=1)
        if vals.ndim == 0:
            vals = vals.reshape(1)
        out[int(m.group(1))] = vals
    return dict(sorted(out.items()))


def load_repeats(metrics_path):
    """{size: repeats} from abft_calibration.csv, so blocks match passes."""
    out = {}
    try:
        with open(metrics_path) as fh:
            for row in csv.DictReader(fh):
                if row.get("calibrate") == "1":
                    out[int(row["M"])] = int(row["repeats"])
    except (OSError, KeyError, ValueError):
        pass
    return out


# ---------------------------------------------------------------------------
# Extreme-value fit
# ---------------------------------------------------------------------------
def block_maxima(vals, repeats):
    """One maximum per calibration pass.

    The dump is an MPI_Gatherv of every rank's readings, so it holds
    repeats * N values in total (each rank contributes repeats * N_b).  We do
    not need to know how many ranks produced it: every reading is an
    independent draw from the same clean-noise distribution, so splitting the
    array into `repeats` contiguous blocks of N gives block maxima over the
    same number of readings a single protected GEMM performs.  Checked against
    a rank-aware reshape: the two agree to within 13%, well inside the
    uncertainty of a 20-sample fit.
    """
    n = len(vals) - (len(vals) % repeats)
    return vals[:n].reshape(repeats, n // repeats).max(axis=1)


def gumbel_fit(maxima):
    """Method of moments.  Returns (mu, beta)."""
    sd = maxima.std(ddof=1)
    beta = sd * math.sqrt(6.0) / math.pi
    if beta <= 0.0:
        beta = float(np.finfo(float).tiny)
    return maxima.mean() - EULER_GAMMA * beta, beta


def fp_probability(tau, mu, beta):
    """P(a clean protected GEMM produces a reading above tau)."""
    z = (tau - mu) / beta
    if z > 700.0:                      # exp(-z) underflows; the answer is exp(-z)
        return math.exp(-z)
    return -math.expm1(-math.exp(-z))  # 1 - exp(-exp(-z)), stable near 0


# ---------------------------------------------------------------------------
# The other side of the trade-off
# ---------------------------------------------------------------------------
def recall_model(size, lo, hi, tau, n=200000, seed=0):
    """P(a single bit flip in bits [lo,hi] moves the element by more than tau).

    A MODEL, not a measurement: A and B are filled from U(-1,1) (see
    fill_random in src/core/common.cuh), so C = A*B is a sum of K products of
    two independent U(-1,1) variables, i.e. approximately N(0, K/9).  Flipping
    one bit of such an element with the same zone ranges the injector uses
    (src/kernels/swifi.cuh) gives the fault magnitude distribution, and a fault
    is detected when it clears tau.

    Validated against the measured per-zone recall: sig_low at 1024 gives
    0.046 modelled vs 0.045 measured, sig_high at 1024 gives 0.94 vs 0.93.
    """
    rng = np.random.default_rng(seed + size)
    c = rng.normal(0.0, math.sqrt(size) / 3.0, n).astype(np.float32)
    bits = rng.integers(lo, hi + 1, n).astype(np.uint32)
    flipped = (c.view(np.uint32) ^ (np.uint32(1) << bits)).view(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        delta = np.abs(flipped.astype(np.float64) - c.astype(np.float64))
    delta = delta[np.isfinite(delta)]
    return float((delta > tau).mean())


def plot_fit(fits, factor, out_path):
    """Block maxima, fitted Gumbel and the chosen threshold, one panel per size.

    The figure carries the argument of the threshold section: the per-pass
    maxima cluster tightly, the fit describes that cluster, and the chosen tau
    sits far to the right of everything the clean runs produced.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(fits)
    ncol = 3
    nrow = (len(sizes) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, size in zip(axes, sizes):
        f = fits[size]
        m, mu, beta = f["maxima"], f["mu"], f["beta"]
        tau = factor * f["observed"]
        lo = min(m.min(), mu - 4 * beta)
        xs = np.linspace(lo, tau * 1.06, 600)
        pdf = np.exp(-(xs - mu) / beta - np.exp(-(xs - mu) / beta)) / beta
        ax.plot(xs, pdf, color="tab:blue", lw=1.6, label="ajuste")
        ax.plot(m, np.zeros_like(m), "|", color="tab:gray", ms=14, mew=1.2,
                label="máximos por pasada")
        ax.axvline(f["observed"], color="tab:gray", ls=":", lw=1.2,
                   label="máximo observado")
        ax.axvline(tau, color="tab:red", ls="--", lw=1.6,
                   label=r"$\tau = %g \times$ máx" % factor)
        ax.set_title(r"$%d^3$" % size, fontsize=10)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for ax in axes[len(sizes):]:
        ax.axis("off")
    axes[0].legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    fig.supxlabel(r"$\max_j |a[j] - e[j]|$ por multiplicación protegida",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if out_path.lower().endswith(".png"):
        fig.savefig(out_path[:-4] + ".eps", format="eps", bbox_inches="tight")
    plt.close(fig)


ZONES = {
    "sign":     (31, 31),
    "exponent": (23, 30),
    "sig_high": (13, 22),
    "sig_low":  (0, 12),
    "any":      (0, 31),
}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diffs-dir", default="docs/calibration",
                    help="directory holding abft_calib_diffs_<N>.csv")
    ap.add_argument("--metrics", default=None,
                    help="abft_calibration.csv (default: <diffs-dir>/abft_calibration.csv)")
    ap.add_argument("--repeats", type=int, default=20,
                    help="calibration passes per size, when --metrics cannot say")
    ap.add_argument("--fp-budget", type=float, default=1e-12,
                    help="max fitted false-positive probability per protected GEMM")
    ap.add_argument("--factor", type=float, default=None,
                    help="force this factor instead of deriving it")
    ap.add_argument("--recall", action="store_true",
                    help="also show modelled recall per zone (see recall_model)")
    ap.add_argument("--emit-table", default=None, help="write a TAU_TABLE bash block here")
    ap.add_argument("--emit-tex", default=None, help="write LaTeX table rows here")
    ap.add_argument("--plot", default=None,
                    help="write the block-maxima / fitted-distribution figure here")
    args = ap.parse_args()

    diffs = load_diffs(args.diffs_dir)
    if not diffs:
        print(f"ERROR: no abft_calib_diffs_*.csv under {args.diffs_dir}", file=sys.stderr)
        return 1

    metrics = args.metrics or os.path.join(args.diffs_dir, "abft_calibration.csv")
    repeats_of = load_repeats(metrics)

    fits = {}
    for size, vals in diffs.items():
        R = repeats_of.get(size, args.repeats)
        if R < 2 or len(vals) < 2 * R:
            print(f"  skipping {size}: needs >= 2 passes, got {R}", file=sys.stderr)
            continue
        maxima = block_maxima(vals, R)
        mu, beta = gumbel_fit(maxima)
        fits[size] = dict(mu=mu, beta=beta, maxima=maxima,
                          observed=float(vals.max()), R=R, n=len(vals))

    # ---- the measurement -------------------------------------------------
    print("Clean-noise calibration — one block maximum per pass")
    print(f"{'N':>7} {'passes':>7} {'readings':>9} {'median':>11} "
          f"{'max/pass':>11} {'max':>11} {'CV(max)':>8}")
    print("-" * 70)
    for size, f in fits.items():
        m = f["maxima"]
        print(f"{size:>7} {f['R']:>7} {f['n']:>9} {np.median(diffs[size]):>11.3e} "
              f"{m.min():>11.3e} {f['observed']:>11.3e} "
              f"{m.std(ddof=1) / m.mean():>8.3f}")

    # ---- the criterion ---------------------------------------------------
    candidates = [1, 2, 3, 4, 5, 10]
    print()
    print("Fitted P(false positive) per protected GEMM, tau = c * max")
    print(f"{'N':>7} " + " ".join(f"{'c=' + str(c):>10}" for c in candidates))
    print("-" * (8 + 11 * len(candidates)))
    worst = {c: 0.0 for c in candidates}
    for size, f in fits.items():
        row = []
        for c in candidates:
            p = fp_probability(c * f["observed"], f["mu"], f["beta"])
            worst[c] = max(worst[c], p)
            row.append(p)
        print(f"{size:>7} " + " ".join(f"{p:>10.1e}" for p in row))
    print(f"{'worst':>7} " + " ".join(f"{worst[c]:>10.1e}" for c in candidates))

    if args.factor is not None:
        factor = args.factor
        why = "forced with --factor"
    else:
        ok = [c for c in candidates if worst[c] <= args.fp_budget]
        if ok:
            factor = min(ok)
            why = (f"smallest integer whose fitted false-positive probability "
                   f"stays <= {args.fp_budget:g} at every size")
        else:
            factor = max(candidates)
            why = (f"NO candidate met the {args.fp_budget:g} budget — "
                   f"falling back to the largest tried")
    print()
    print(f"==> safety factor = {factor:g}   ({why})")
    print(f"    worst-case fitted false-positive rate: "
          f"{worst.get(factor, float('nan')):.1e} per protected GEMM")

    # ---- what the factor costs ------------------------------------------
    if args.recall:
        print()
        print("Modelled recall per zone (higher factor => more false negatives)")
        zone_names = ["sign", "exponent", "sig_high", "sig_low", "any"]
        print(f"{'N':>7} {'zone':>9} " + " ".join(f"{'c=' + str(c):>8}" for c in candidates))
        print("-" * (18 + 9 * len(candidates)))
        for size, f in fits.items():
            for z in zone_names:
                lo, hi = ZONES[z]
                row = [recall_model(size, lo, hi, c * f["observed"]) for c in candidates]
                print(f"{size:>7} {z:>9} " + " ".join(f"{r:>8.3f}" for r in row))
            print()

    # ---- outputs ---------------------------------------------------------
    if args.plot:
        try:
            plot_fit(fits, factor, args.plot)
            print(f"\nwrote {args.plot}")
        except ImportError:
            print("\n(figure skipped: matplotlib not available)")

    if args.emit_table:
        with open(args.emit_table, "w") as fh:
            fh.write("# Generated by scripts/analyze_calibration.py\n")
            fh.write(f"# tau = {factor:g} x observed max |a-e|\n")
            fh.write(f"# factor: {why}\n")
            fh.write("declare -A TAU_TABLE=(\n")
            for size, f in fits.items():
                fh.write(f'  [{size}]="{factor * f["observed"]:.6f}"\n')
            fh.write(")\n")
        print(f"\nwrote {args.emit_table}")

    if args.emit_tex:
        with open(args.emit_tex, "w") as fh:
            fh.write("% Generated by scripts/analyze_calibration.py\n")
            fh.write(f"% tau = {factor:g} x observed max |a-e|; {why}\n")
            for size, f in fits.items():
                p = fp_probability(factor * f["observed"], f["mu"], f["beta"])
                fh.write("$%d^3$ & $%.2e$ & $%g$ & $%.2e$ & $%.0e$ \\\\\n"
                         % (size, f["observed"], factor, factor * f["observed"], p))
        print(f"wrote {args.emit_tex}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
