#pragma once

#include "../core/common.cuh"
#include "../core/types.cuh"

// ===================================================================
// SWIFI — Software-Implemented Fault Injection
// ===================================================================
// Injects a single bit-flip into one element of a C fragment.  The bit
// position is uniformly random within a ZONE of the IEEE-754 word:
//
//   IEEE-754 float32:  [31]=sign  [30..23]=exponent  [22..0]=significand
//
//   zone "any"      -> bits [0,31]   (whole word; the default campaign)
//   zone "sign"     -> bit  31       (huge Δ)
//   zone "exponent" -> bits [23,30]  (Δ ~ 2^e, huge)
//   zone "sig_high" -> bits [13,22]  (top 10 significand bits; moderate/large Δ)
//   zone "sig_low"  -> bits [0,12]   (bottom 13 significand bits; tiny Δ → FN-prone)
//
// Restricting the zone lets the campaign measure recall PER bit region,
// which is what explains the false negatives (sig_low flips are below τ).
// ===================================================================

// Map a zone name to an inclusive bit range [lo, hi].
inline void swifi_zone_range(const std::string& zone, int& lo, int& hi) {
    if      (zone == "sign")     { lo = 31; hi = 31; }
    else if (zone == "exponent") { lo = 23; hi = 30; }
    else if (zone == "sig_high") { lo = 13; hi = 22; }
    else if (zone == "sig_low")  { lo =  0; hi = 12; }
    else                         { lo =  0; hi = 31; }   // "any" / unknown
}

__global__ inline void k_inject_bitflip(float* C, int ldc,
                                        int target_row, int target_col,
                                        int bit_position) {
    unsigned int* ptr = reinterpret_cast<unsigned int*>(&C[target_row * ldc + target_col]);
    unsigned int val = *ptr;
    val ^= (1u << bit_position);
    *ptr = val;
}

// ===================================================================
// Additive fault (NOT a real SWIFI bit-flip) — OVERHEAD studies only.
// Adds a large constant to one random element of the fragment BEFORE
// verification, so it ALWAYS exceeds the detection threshold (the analog
// of the reference project's `res[0] += 10000`).  Isolates the ABFT
// detect+localize+correct *overhead* from the SWIFI RNG/zone machinery.
// Use SWIFI (above) for accuracy/recall — never this.
// 1e6 >> every threshold in these experiments (formula τ≈O(10²) @4096³,
// comparison --threshold 1.0, calibrated campaign τ) so it is detected
// at any size, unlike a literal "1000" a huge matrix's noise floor could
// approach.
constexpr float ADD_FAULT_DELTA = 1.0e6f;

__global__ inline void k_inject_add(float* C, int ldc,
                                    int target_row, int target_col,
                                    float delta) {
    C[target_row * ldc + target_col] += delta;
}

inline InjectionInfo inject_add_constant(float* dC_frag, int ldc,
                                         int M_b, int N_frag, int frag_idx,
                                         uint64_t seed, cudaStream_t stream) {
    InjectionInfo info{};
    info.injected   = true;
    info.frag_index = frag_idx;
    std::mt19937_64 gen(seed + static_cast<uint64_t>(frag_idx));
    std::uniform_int_distribution<int> row_dist(0, M_b - 1);
    std::uniform_int_distribution<int> col_dist(0, N_frag - 1);
    info.row          = row_dist(gen);
    info.col          = col_dist(gen);
    info.bit_position = -1;
    k_inject_add<<<1, 1, 0, stream>>>(
        dC_frag, ldc, info.row, info.col, ADD_FAULT_DELTA);
    return info;
}

inline InjectionInfo inject_single_bitflip(float* dC_frag, int ldc,
                                           int M_b, int N_frag,
                                           int frag_idx,
                                           uint64_t seed,
                                           cudaStream_t stream,
                                           const std::string& zone = "any") {
    InjectionInfo info{};
    info.injected   = true;
    info.frag_index = frag_idx;

    int bit_lo, bit_hi;
    swifi_zone_range(zone, bit_lo, bit_hi);

    std::mt19937_64 gen(seed + static_cast<uint64_t>(frag_idx));
    std::uniform_int_distribution<int> row_dist(0, M_b - 1);
    std::uniform_int_distribution<int> col_dist(0, N_frag - 1);
    std::uniform_int_distribution<int> bit_dist(bit_lo, bit_hi);

    info.row          = row_dist(gen);
    info.col          = col_dist(gen);
    info.bit_position = bit_dist(gen);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&info.value_before,
                          &dC_frag[info.row * ldc + info.col],
                          sizeof(float), cudaMemcpyDeviceToHost));

    k_inject_bitflip<<<1, 1, 0, stream>>>(
        dC_frag, ldc, info.row, info.col, info.bit_position);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&info.value_after,
                          &dC_frag[info.row * ldc + info.col],
                          sizeof(float), cudaMemcpyDeviceToHost));

    return info;
}
