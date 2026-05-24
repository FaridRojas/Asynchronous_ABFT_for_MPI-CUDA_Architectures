#pragma once

#include "common.cuh"
#include "types.cuh"

// ---------------------------------------------------------------------------
// Help text
// ---------------------------------------------------------------------------
inline void print_help() {
    std::cout <<
"Usage:\n"
"  abft_gemm <M> <K> <N> [flags]            rectangular form\n"
"  abft_gemm <N>         [flags]            square shorthand (M = K = N)\n"
"\n"
"Flags:\n"
"  --inject {none|swifi|add}      none; SWIFI single-bit-flip (accuracy);\n"
"                                 add = large additive fault, always >tau\n"
"                                 (OVERHEAD studies only — not real SWIFI)\n"
"  --swifi-zone {any|sign|exponent|sig_high|sig_low}\n"
"                                 restrict the flipped bit to a region of the\n"
"                                 IEEE-754 word (default any)\n"
"  --baseline                     run only the unprotected GEMM\n"
"  --calibrate                    run calibration: report max |actual-expected|\n"
"                                 and a suggested threshold; no detection logic\n"
"  --threshold T                  override detection threshold with T (>0).\n"
"                                 Use the value reported by --calibrate.\n"
"  --calibration-safety F         multiplier on observed max for the suggestion\n"
"                                 (default 10)\n"
"  --frags-per-rank F             pipeline depth per rank (default 4)\n"
"  --frag-cap E                   resilience: cap elements per fragment so a\n"
"                                 huge stripe gets MORE fragments (raises F to\n"
"                                 ceil(M_b*N_b/E)); 0=off (default).  Keeps the\n"
"                                 <=1-fault-per-fragment assumption at >10k\n"
"  --grid Pr Pc                   2D process grid (default: auto near-square)\n"
"  --repeats N                    iters per timed trial (default 20).  Set to 1\n"
"                                 in COMPARISON mode (with --reseed-per-trial)\n"
"  --warmups N                    discarded warm-up trials per phase (default 2;\n"
"                                 raise for tiny matrices to kill clock jitter)\n"
"  --samples N                    outer trial count (default 5).  COMPARISON mode\n"
"                                 uses 20, CONFUSION-MATRIX campaign uses 100\n"
"  --reseed-per-trial             regenerate A,B (and recompute golden) before\n"
"                                 each timed trial so every sample is independent\n"
"  --seed-a S, --seed-b S         RNG seeds for A and B\n"
"  --csv PATH                     metrics CSV output path (default abft_metrics.csv)\n"
"  --calib-diffs PATH             where to dump every observed |actual-expected|\n"
"                                 (one per line) during --calibrate\n"
"                                 (default abft_calibration_diffs.csv)\n"
"  --help                         this message\n";
}

// ---------------------------------------------------------------------------
// CLI parser
// ---------------------------------------------------------------------------
inline ExperimentConfig parse_args(int argc, char** argv) {
    ExperimentConfig c{};
    std::vector<int> positional;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--inject"             && i + 1 < argc) c.inject        = argv[++i];
        else if (a == "--swifi-zone"         && i + 1 < argc) c.swifi_zone    = argv[++i];
        else if (a == "--frag-cap"           && i + 1 < argc) c.frag_cap      = std::atoi(argv[++i]);
        else if (a == "--baseline")                           c.baseline_only = true;
        else if (a == "--calibrate")                          c.calibrate     = true;
        else if (a == "--threshold"          && i + 1 < argc) c.threshold_override = std::atof(argv[++i]);
        else if (a == "--calibration-safety" && i + 1 < argc) c.calibration_safety_factor = std::atof(argv[++i]);
        else if (a == "--frags-per-rank"     && i + 1 < argc) c.frags_per_rank = std::atoi(argv[++i]);
        else if (a == "--grid"               && i + 2 < argc) { c.Pr = std::atoi(argv[++i]); c.Pc = std::atoi(argv[++i]); }
        else if (a == "--repeats"            && i + 1 < argc) c.repeats       = std::atoi(argv[++i]);
        else if (a == "--warmups"            && i + 1 < argc) c.warmups       = std::atoi(argv[++i]);
        else if (a == "--samples"            && i + 1 < argc) c.num_samples   = std::atoi(argv[++i]);
        else if (a == "--reseed-per-trial")                   c.reseed_per_trial = true;
        else if (a == "--seed-a"             && i + 1 < argc) c.seed_a        = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--seed-b"             && i + 1 < argc) c.seed_b        = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--csv"                && i + 1 < argc) c.csv_path        = argv[++i];
        else if (a == "--calib-diffs"        && i + 1 < argc) c.calib_diffs_path= argv[++i];
        else if (a == "--help"               || a == "-h")   { print_help(); std::exit(0); }
        else if (!a.empty() && a[0] != '-')                   positional.push_back(std::atoi(argv[i]));
    }

    if      (positional.size() == 1) { c.M = c.K = c.N = positional[0]; }
    else if (positional.size() >= 3) { c.M = positional[0]; c.K = positional[1]; c.N = positional[2]; }

    if (c.frags_per_rank < 1) c.frags_per_rank = 1;
    if (c.repeats        < 1) c.repeats        = 1;
    if (c.warmups        < 0) c.warmups        = 0;
    if (c.frag_cap       < 0) c.frag_cap       = 0;
    if (c.num_samples    < 0) c.num_samples    = 0;
    return c;
}
