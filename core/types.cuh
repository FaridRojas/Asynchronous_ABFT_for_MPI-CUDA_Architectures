#pragma once

#include "common.cuh"

// 2D process grid
struct Grid2D {
    int Pr;          // number of process rows
    int Pc;          // number of process columns
    int pr;          // this rank's row coordinate
    int pc;          // this rank's column coordinate
    MPI_Comm row_comm;   // ranks sharing the same pr (i.e., same A-stripe)
    MPI_Comm col_comm;   // ranks sharing the same pc (i.e., same B-stripe)
};

// Result of one ABFT verification on a fragment
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

// CLI configuration for one experiment run
struct ExperimentConfig {
    int         M = 1024;
    int         K = 1024;
    int         N = 1024;
    int         Pr = 0, Pc = 0;       // 0 = auto

    int         frags_per_rank = 4;
    int         repeats        = 20;  // timing repetitions / SWIFI trials
    // / Discarded warm-up trials before the timed ones, run at the REAL
    int         warmups        = 2;
    // / RESILIENCE knob (default 0 = off).  Caps elements per fragment:
    int         frag_cap       = 0;

    // / Outer trial count.  0 = legacy default (5).  In COMPARISON mode set
    int         num_samples    = 0;

    // / When true, A and B (and their golden C when needed) are regenerated
    bool        reseed_per_trial = false;

    // / Only "online" is supported now (verification concurrent with cuBLAS).
    std::string scheme         = "online";
    std::string inject         = "none";    // "none" or "swifi"
    // / Restrict the SWIFI bit-flip to a region of the IEEE-754 word:
    std::string swifi_zone     = "any";
    bool        baseline_only  = false;
    bool        calibrate      = false;     // run calibration pass instead of normal pass

    // / User-supplied detection threshold.  If > 0, overrides the formula and
    double      threshold_override = -1.0;

    // / Multiplier applied to the observed max |actual-expected| in calibration
    double      calibration_safety_factor = 10.0;

    uint64_t    seed_a = 123456;
    uint64_t    seed_b = 987654;

    std::string csv_path        = "abft_metrics.csv";
    // / Where calibration writes one |actualRow-expectedRow| value per line
    std::string calib_diffs_path = "abft_calibration_diffs.csv";
};

// SWIFI bookkeeping for one injection
struct InjectionInfo {
    bool  injected     = false;
    int   frag_index   = -1;
    int   row          = -1;
    int   col          = -1;
    int   bit_position = -1;
    float value_before = 0.0f;
    float value_after  = 0.0f;
};
