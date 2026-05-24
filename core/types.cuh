#pragma once

#include "common.cuh"

// ---------------------------------------------------------------------------
// 2D process grid
// ---------------------------------------------------------------------------
struct Grid2D {
    int Pr;          // number of process rows
    int Pc;          // number of process columns
    int pr;          // this rank's row coordinate
    int pc;          // this rank's column coordinate
    MPI_Comm row_comm;   // ranks sharing the same pr (i.e., same A-stripe)
    MPI_Comm col_comm;   // ranks sharing the same pc (i.e., same B-stripe)
};

// ---------------------------------------------------------------------------
// Result of one ABFT verification on a fragment
// ---------------------------------------------------------------------------
struct ABFTResult {
    bool  error_detected   = false;
    bool  error_corrected  = false;
    int   frag_index       = -1;
    int   error_row        = -1;     // row inside the fragment (0..M_b-1)
    int   error_col        = -1;     // column inside the fragment (0..N_frag-1)
    float corrupted_value  = 0.0f;
    float corrected_value  = 0.0f;
    float golden_value     = 0.0f;   // filled by orchestrator if known
};

// ---------------------------------------------------------------------------
// CLI configuration for one experiment run
// ---------------------------------------------------------------------------
struct ExperimentConfig {
    int         M = 1024;
    int         K = 1024;
    int         N = 1024;
    int         Pr = 0, Pc = 0;       // 0 = auto

    int         frags_per_rank = 4;
    int         repeats        = 20;  // timing repetitions / SWIFI trials
    /// Discarded warm-up trials before the timed ones, run at the REAL
    /// problem shape (per phase: baseline and protected).  Tiny matrices
    /// need many — their compute time is dwarfed by launch/clock jitter,
    /// which is what makes their measured overhead noisy / negative.
    int         warmups        = 2;
    /// RESILIENCE knob (default 0 = off).  Caps elements per fragment:
    /// when set, F is raised to ceil(M_b*N_b / frag_cap) so a very large
    /// stripe (>10k, 20k, 100k) is split into MORE, smaller fragments,
    /// keeping the ABFT "<=1 fault per fragment" assumption valid.
    /// This is the opposite of a perf knob — it adds fragments.
    int         frag_cap       = 0;

    /// Outer trial count.  0 = legacy default (5).  In COMPARISON mode set
    /// this to 20 (small grid) or whatever the sbatch passes, combined
    /// with --repeats 1 + --reseed-per-trial for 20 fresh-RNG samples.
    int         num_samples    = 0;

    /// When true, A and B (and their golden C when needed) are regenerated
    /// from a fresh seed BEFORE each timed trial — so each "sample" is an
    /// independent measurement on a different random matrix pair.  The
    /// CONFUSION-MATRIX campaign uses this with --samples 100 to build a
    /// robust per-shape CM; the comparison sweep uses --samples 20.
    bool        reseed_per_trial = false;

    /// Only "online" is supported now (verification concurrent with cuBLAS).
    /// "baseline" is selected by --baseline.  Post-hoc has been removed.
    std::string scheme         = "online";
    std::string inject         = "none";    // "none" or "swifi"
    /// Restrict the SWIFI bit-flip to a region of the IEEE-754 word:
    ///   any | sign | exponent | sig_high | sig_low
    /// Used to characterise detection (recall) per bit region.
    std::string swifi_zone     = "any";
    bool        baseline_only  = false;
    bool        calibrate      = false;     // run calibration pass instead of normal pass

    /// User-supplied detection threshold.  If > 0, overrides the formula and
    /// is used uniformly for every fragment.  Set this to the value printed
    /// by a previous `--calibrate` run.
    double      threshold_override = -1.0;

    /// Multiplier applied to the observed max |actual-expected| in calibration
    /// to recommend a working threshold.
    double      calibration_safety_factor = 10.0;

    uint64_t    seed_a = 123456;
    uint64_t    seed_b = 987654;

    std::string csv_path        = "abft_metrics.csv";
    /// Where calibration writes one |actualRow-expectedRow| value per line
    /// for downstream histogram / CDF plotting.  Only written when
    /// `--calibrate` is set.
    std::string calib_diffs_path = "abft_calibration_diffs.csv";
};

// ---------------------------------------------------------------------------
// SWIFI bookkeeping for one injection
// ---------------------------------------------------------------------------
struct InjectionInfo {
    bool  injected     = false;
    int   frag_index   = -1;
    int   row          = -1;
    int   col          = -1;
    int   bit_position = -1;
    float value_before = 0.0f;
    float value_after  = 0.0f;
};
