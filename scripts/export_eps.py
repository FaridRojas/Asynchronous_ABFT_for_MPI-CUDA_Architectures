#!/usr/bin/env python3
"""export_eps.py — re-render every data plot as vector EPS for LaTeX.

The existing plotters (plot_comparison / plot_metrics / plot_real_gflops /
plot_zone_recall) all call ``fig.savefig(<name>.png, dpi=150)``.  Rather
than duplicate any plotting logic, this script monkey-patches
``matplotlib.figure.Figure.savefig`` so that every ``*.png`` written is
ALSO written as a sibling ``*.eps`` (true vector, editable fonts), then
drives each plotter's ``main()`` on whatever result data is present
under ``docs/``.  PNGs are still produced unchanged.

Usage:
    python3 scripts/export_eps.py [--root REPO_ROOT]

EPS files land next to their PNGs (docs/comparison*/, docs/campaign/,
docs/calibration*/, docs/profile/).  Include them in LaTeX with
\\includegraphics{...} (no extension) — pdflatex picks the PDF it
auto-generates, latex/dvips picks the EPS.
"""
import argparse
import glob
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
# Type-42 (TrueType) fonts embed editable text in the EPS instead of
# the default Type-3 bitmaps — sharper in the printed paper.
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.figure as _mfig  # noqa: E402

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))

# --- savefig shim: every PNG also emitted as vector EPS ---------------
_orig_savefig = _mfig.Figure.savefig


def _savefig_also_eps(self, fname, *args, **kwargs):
    _orig_savefig(self, fname, *args, **kwargs)          # keep the PNG
    s = str(fname)
    if s.lower().endswith(".png"):
        eps = s[:-4] + ".eps"
        k = dict(kwargs)
        k.pop("dpi", None)                               # EPS is vector
        try:
            _orig_savefig(self, eps, *args, **k)
            print(f"    + {os.path.relpath(eps)}")
        except Exception as exc:                          # noqa: BLE001
            print(f"    ! EPS failed for {eps}: {exc}", file=sys.stderr)


_mfig.Figure.savefig = _savefig_also_eps


def _load(mod_name):
    path = os.path.join(_SCRIPTS, mod_name + ".py")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod_name, argv, what):
    """Invoke a plotter's main() with a synthetic argv; tolerate the
    SystemExit/`sys.exit` the plotters raise on missing inputs."""
    print(f"==> {what}")
    mod = _load(mod_name)
    old = sys.argv
    sys.argv = [mod_name + ".py"] + argv
    try:
        mod.main()
    except SystemExit as e:                               # plotter bailed
        if e.code not in (0, None):
            print(f"    (skipped: {mod_name} exited {e.code})")
    except Exception as exc:                               # noqa: BLE001
        print(f"    ! {mod_name} failed: {exc}", file=sys.stderr)
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(_SCRIPTS),
                    help="repository root (default: parent of scripts/)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    docs = os.path.join(root, "docs")
    os.chdir(root)

    # 1) Per-regime GFLOPS / overhead comparison(s)
    for csv in sorted(glob.glob(os.path.join(docs, "comparison*",
                                             "compare_all.csv"))):
        _run("plot_comparison",
             ["--csv", csv, "--outdir", os.path.dirname(csv)],
             f"comparison  {os.path.relpath(csv)}")

    # 2) Felix campaign (timing / gflops / overhead / detection / CM)
    for d in sorted(glob.glob(os.path.join(docs, "campaign*"))) + \
             sorted(glob.glob(os.path.join(docs, "calibration*"))):
        if not os.path.isdir(d):
            continue
        # Where the metrics CSV lives depends on who wrote it.  The cluster
        # job renders straight into scripts/plots/<dir>/, but download.sh
        # splits the results: PNGs land in docs/<dir>/ while the CSV stays at
        # docs/abft_metrics[_<suffix>].csv.  Looking only inside <dir> made
        # every campaign silently skip (the `continue` below), so the figures
        # under docs/campaign*/ stayed whatever the cluster had rendered and
        # never picked up a local plot fix.
        suffix = os.path.basename(d)[len("campaign"):] \
                 if os.path.basename(d).startswith("campaign") else ""
        candidates = [os.path.join(d, "abft_metrics.csv"),
                      os.path.join(d, "abft_calibration.csv"),
                      os.path.join(docs, f"abft_metrics{suffix}.csv")]
        mcsv = next((c for c in candidates if os.path.exists(c)), None)
        if mcsv is None:
            print(f"    (skipped {os.path.relpath(d)}: no metrics CSV in "
                  + ", ".join(os.path.relpath(c) for c in candidates) + ")")
            continue
        argv = ["--metrics", mcsv, "--out", d]
        diffs = glob.glob(os.path.join(d, "abft_calib*diffs*.csv"))
        if diffs:
            argv += ["--diffs", diffs[0]]
        _run("plot_metrics", argv, f"campaign    {os.path.relpath(mcsv)}")

    # 3) Per-zone recall sweep
    for csv in sorted(glob.glob(os.path.join(docs, "calibration*",
                                             "abft_zone_recall.csv"))):
        _run("plot_zone_recall",
             ["--csv", csv,
              "--out", os.path.join(os.path.dirname(csv),
                                    "zone_recall.png")],
             f"zone-recall {os.path.relpath(csv)}")

    # 4) Measured (real) GFLOPS — ours vs theirs, per regime.
    #    felix profiling -> nvprof (*_flops.csv); pacca -> ncu (*.ncu-rep).
    #    Both land in docs/profile/; detect each independently.
    prof = os.path.join(docs, "profile")
    if glob.glob(os.path.join(prof, "*_flops.csv")):
        _run("plot_real_gflops",
             ["--indir", prof, "--tool", "nvprof",
              "--out", os.path.join(prof, "real_gflops_felix.png")],
             f"real-gflops {os.path.relpath(prof)} (nvprof)")
    if glob.glob(os.path.join(prof, "*.ncu-rep")):
        _run("plot_real_gflops",
             ["--indir", prof, "--tool", "ncu",
              "--out", os.path.join(prof, "real_gflops_pacca.png")],
             f"real-gflops {os.path.relpath(prof)} (ncu)")
    # 4b) Real GFLOPS vs size (ncu sweep) — line plot
    for sweep in glob.glob(os.path.join(docs, "profile*",
                                        "real_gflops_sweep.csv")):
        _run("plot_real_gflops",
             ["--sweep-csv", sweep,
              "--out", os.path.join(os.path.dirname(sweep),
                                    "real_gflops_vs_size_pacca.png")],
             f"real-gflops-vs-size {os.path.relpath(sweep)}")

    print("Done. EPS files written alongside the PNGs.")


if __name__ == "__main__":
    main()
