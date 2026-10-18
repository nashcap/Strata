// src/kernels/cpu/iq_avx1_parity.cpp - the AVX1 expert kernels (iq_avx1.cpp) against ggml-cpu's dots and
// against themselves at every token count.  No GPU, no model: random blocks (every grid index and sign
// pattern occurs) and random activations quantized the way the engine quantizes them.
//
//     iq_avx1_parity            IQ3_XXS/IQ3_S/IQ2_S/IQ4_XS gate/up rows and IQ4_NL/Q2_0 down rows:
//                               every row bit for bit the same kernel's one-token row (#152), and rel <= 1e-5
//                               against ggml-cpu's vec_dot (the scalar _generic the engine runs without these).
//     iq_avx1_parity --bench [--mb 128] [--nt 1,2,4] [--cpu N]
//                               ms per expert on one thread: ggml-cpu's per-token dot vs the iq128 kernel over
//                               --mb MB of expert blobs (more than any L3: the weights stream from DRAM as in
//                               decode), for the nt in --nt.
//
// Runs on any CPU with AVX (the kernels are compiled for SSE4.2 + AVX); on an AVX2 machine it checks the
// older-CPU path exactly as an AVX-only machine runs it (the engine's dispatch gate is separate: this calls
// the kernels directly).
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sched.h>
#endif

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kH = 2560, kFF = 640;   // the experts' geometry: n_embd, expert width (as iq_avx2_parity)
constexpr int kMaxT = 8;

uint64_t mix(uint64_t& s) {   // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

const ggml_type_traits_cpu* traits_cpu(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

// random bytes, then a sane fp16 scale at the front of every block (iq_avx2_parity's)
void fill_blocks(uint8_t* p, size_t bytes, int type, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) {
        const uint64_t v = mix(seed);
        std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i));
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const float d = (0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * 1e-3f;
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(p + o, &h, 2);
    }
}

void fill_blob(uint8_t* blob, const cpu::NativeFmt& f, uint64_t& seed) {
    fill_blocks(blob, f.down_off, f.gu_type, seed);                        // the gate and up rows
    fill_blocks(blob + f.down_off, f.bytes - f.down_off, f.d_type, seed);  // the down rows
}

struct Case {
    int gu, dn;
    const char* name;
};

const Case kCases[] = {
    {18, 20, "iq3_xxs/iq4_nl"}, {21, 20, "iq3_s/iq4_nl"}, {22, 20, "iq2_s/iq4_nl"}, {23, 20, "iq4_xs/iq4_nl"},
    {18, 42, "iq3_xxs/q2_0"},   {21, 42, "iq3_s/q2_0"},   {22, 42, "iq2_s/q2_0"},   {23, 42, "iq4_xs/q2_0"},
};

// one expert, nt tokens of quantized activations, and the reference rows
struct Fixture {
    cpu::NativeFmt f;
    std::vector<uint8_t> blob;
    std::vector<std::vector<uint8_t>> act, hq;          // per token
    std::vector<std::vector<float>> x, h;
    std::vector<float> ff_ref, out_ref;                 // [t][row]
};

void make_fixture(Fixture& fx, int gu, int dn, int nt, uint64_t& seed) {
    std::string err;
    if (!cpu::native_fmt(gu, dn, kH, kFF, fx.f, err)) { std::fprintf(stderr, "native_fmt: %s\n", err.c_str()); std::exit(1); }
    fx.blob.resize(fx.f.bytes);
    fill_blob(fx.blob.data(), fx.f, seed);
    fx.act.resize(nt); fx.hq.resize(nt); fx.x.resize(nt); fx.h.resize(nt);
    for (int t = 0; t < nt; ++t) {
        fx.act[t].resize(fx.f.act_bytes); fx.hq[t].resize(fx.f.h_bytes);
        fx.x[t].resize(kH); fx.h[t].resize(kFF);
        for (int i = 0; i < kH; ++i) fx.x[t][i] = (float) ((mix(seed) >> 40) * 0x1p-24 * 2.0 - 1.0);
        for (int i = 0; i < kFF; ++i) fx.h[t][i] = (float) ((mix(seed) >> 40) * 0x1p-24 * 2.0 - 1.0);
        cpu::native_quant_act(fx.f, fx.x[t].data(), fx.act[t].data());
        cpu::native_quant_h(fx.f, fx.h[t].data(), fx.hq[t].data());
    }
    // reference rows: ggml-cpu's own dots, per token (what the engine runs without the iq128 kernels)
    const ggml_vec_dot_t gu_dot = traits_cpu(gu)->vec_dot;
    const ggml_vec_dot_t dn_dot = traits_cpu(dn)->vec_dot;
    fx.ff_ref.resize((size_t) nt * kFF); fx.out_ref.resize((size_t) nt * kH);
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < kFF; ++r) {
            float g = 0.f, u = 0.f;
            gu_dot(kH, &g, 0, fx.blob.data() + (size_t) r * fx.f.gu_row, 0, fx.act[t].data(), 0, 1);
            gu_dot(kH, &u, 0, fx.blob.data() + fx.f.up_off + (size_t) r * fx.f.gu_row, 0, fx.act[t].data(), 0, 1);
            fx.ff_ref[(size_t) t * kFF + r] = (g / (1.f + std::exp(-g))) * u;
        }
        for (int r = 0; r < kH; ++r) {
            float s = 0.f;
            dn_dot(kFF, &s, 0, fx.blob.data() + fx.f.down_off + (size_t) r * fx.f.d_row, 0, fx.hq[t].data(), 0, 1);
            fx.out_ref[(size_t) t * kH + r] = s;
        }
    }
}

int failures = 0;

// FNV-1a over every row any kernel produced in this run.  The checks above are self-consistency (the same
// kernel at nt and at nt=1) plus rel <= 1e-5 against ggml-cpu - they do not say a new kernel agrees with the
// kernel it replaces to the bit, and the seeds are fixed, so two builds printing the same fingerprint do.
uint64_t row_fp = 1469598103934665603ull;
std::map<std::string, uint64_t> rung_fp;   // per format+side, so a change can be pinned to one rung
std::map<std::string, std::pair<uint64_t, double>> vs_ggml;   // per rung: rows differing bitwise, worst rel
uint64_t fp_of(const char* s, const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    for (const char* p = s; *p; ++p) { h ^= (unsigned char) *p; h *= 1099511628211ull; }
    for (float f : v) {
        uint32_t b;
        std::memcpy(&b, &f, sizeof b);
        h ^= b & 0xFF; h *= 1099511628211ull;
        h ^= (b >> 8) & 0xFF; h *= 1099511628211ull;
        h ^= (b >> 16) & 0xFF; h *= 1099511628211ull;
        h ^= (b >> 24) & 0xFF; h *= 1099511628211ull;
    }
    return h;
}

void check(const char* what, const std::vector<float>& got, const std::vector<float>& ref, bool exact) {
    const uint64_t h = fp_of(what, got);
    row_fp ^= h; row_fp *= 1099511628211ull;
    std::string key = std::string(what);
    size_t sp = key.find(" nt=");
    if (sp != std::string::npos) key.erase(sp);
    rung_fp[key] ^= h;
    if (exact) {   // bit-exact (iq_avx2_parity's rows_differ)
        size_t d = 0;
        for (size_t i = 0; i < ref.size(); ++i) d += std::memcmp(&got[i], &ref[i], sizeof(float)) != 0;
        if (d) { std::printf("  FAIL %s: %zu of %zu rows differ\n", what, d, ref.size()); ++failures; }
        return;
    }
    // the repo's bar (iq_avx2_parity's rel): aggregate sum|diff| / sum|ref| <= 1e-5
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += std::fabs((double) got[i] - (double) ref[i]);
        den += std::fabs((double) ref[i]);
    }
    const double r = num / (den + 1e-30);
    size_t diff = 0;
    for (size_t i = 0; i < ref.size(); ++i) diff += std::memcmp(&got[i], &ref[i], sizeof(float)) != 0;
    std::string k = std::string(what);
    size_t kp = k.find(" nt=");
    if (kp != std::string::npos) k.erase(kp);
    vs_ggml[k].first += diff;
    if (r > vs_ggml[k].second) vs_ggml[k].second = r;
    if (!(r <= 1e-5)) { std::printf("  FAIL %s: rel %.3e > 1e-5 vs ggml-cpu\n", what, r); ++failures; }
}

// ---- PROTOTYPE (benchmark only, not the engine): IQ2_S with the 10-bit grid indices pre-merged into
// 16-bit words, so the decode no longer builds them from qs bytes + packed qh bits.
// If this wins at nt=1 the real version is an offline repack in the pack (tools/iq_pack.py) plus a second
// gate/up blob pointer in NativeFmt - NOT a runtime repack cache. The GPU tier keeps reading the GGUF.
// shortcut: duplicates the float stage of iq_avx1.cpp's row_dot_r; delete it once the engine has a repacked rung.
#include <immintrin.h>

struct Rp22 {                                    // one repacked ggml block (256 values), 106 bytes
    uint8_t  d[2];                               // 2
    uint16_t idx[32];                            // 64: 8 groups x 4 grid indices (qs bits | 2 qh bits)
    uint8_t  sign[32];                           // 32
    uint8_t  scale[8];                           // 8
};
static_assert(sizeof(Rp22) == 106, "repacked IQ2_S block must be 106 bytes");

struct RpSign8 { uint64_t v[256]; RpSign8() : v() {
    for (unsigned i = 0; i < 256; ++i) { uint64_t w = 0;
        for (int k = 0; k < 8; ++k) w |= (uint64_t) (((i >> k) & 1u) ? 0xFFu : 1u) << (8 * k);
        v[i] = w; } } };
struct RpK16 { alignas(16) int16_t v[16][16]; RpK16() : v() {
    for (int k = 0; k < 16; ++k) for (int l = 0; l < 16; ++l) v[k][l] = (int16_t) (2 * k + 1); } };
static const RpSign8 rp_sign8{};
static const RpK16 rp_k16{};

void rp22_block(const uint8_t* src, Rp22* dst) {
    std::memcpy(dst->d, src, 2);
    const uint8_t* qs = src + 2;                       // 32 grid index bytes
    const uint8_t* qh = src + 66;                      // 8 bytes, 2 high bits per index
    for (int h = 0; h < 8; ++h) {
        const int j = h >> 1, half = h & 1;
        const uint8_t hq = qh[2 * j + half];
        const uint8_t* q = qs + 8 * j + 4 * half;
        dst->idx[4 * h + 0] = (uint16_t) (q[0] | ((hq & 0x03) << 8));
        dst->idx[4 * h + 1] = (uint16_t) (q[1] | ((hq & 0x0C) << 6));
        dst->idx[4 * h + 2] = (uint16_t) (q[2] | ((hq & 0x30) << 4));
        dst->idx[4 * h + 3] = (uint16_t) (q[3] | ((hq & 0xC0) << 2));
    }
    std::memcpy(dst->sign,  src + 34, 32);
    std::memcpy(dst->scale, src + 74, 8);
}

// fp16 -> fp32, the same bits as iq_avx1.cpp's h2f (fp16 -> fp32 is exact, so any correct conversion matches)
float h2f_local(uint16_t h) {
    const uint32_t e = h & 0x7C00u;
    if (e && e != 0x7C00u) {
        const uint32_t f = ((uint32_t) (h & 0x7FFFu) << 13) + 0x38000000u | (uint32_t) (h & 0x8000u) << 16;
        float out; std::memcpy(&out, &f, 4); return out;
    }
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) { f = sign << 31; }
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) { f = (sign << 31) | 0x7F800000u | (man << 13); }
    else { f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13); }
    float out; std::memcpy(&out, &f, 4); return out;
}

// one repacked row, the mirror of iq_avx1.cpp's row_dot_r<22, NT, 1> (same accumulators, same order):
// gate and up are separate calls here exactly as gu_rows does them, so register pressure matches production
template <int NT>
void rp_row_dot(const Rp22* row, int nb, const block_q8_K* const* y, float* res) {
    __m128 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        __m128i a0[NT], a1[NT];
        for (int t = 0; t < NT; ++t) a0[t] = a1[t] = _mm_setzero_si128();
        const Rp22* blk = row + i;
        for (int h = 0; h < 8; ++h) {
            const uint16_t* ii = blk->idx + 4 * h;
            const __m128i g0 = _mm_set_epi64x((long long) iq2s_grid[ii[1]], (long long) iq2s_grid[ii[0]]);
            const __m128i g1 = _mm_set_epi64x((long long) iq2s_grid[ii[3]], (long long) iq2s_grid[ii[2]]);
            const uint8_t* sp = blk->sign + 4 * h;
            const __m128i s0 = _mm_set_epi64x((long long) rp_sign8.v[sp[1]], (long long) rp_sign8.v[sp[0]]);
            const __m128i s1 = _mm_set_epi64x((long long) rp_sign8.v[sp[3]], (long long) rp_sign8.v[sp[2]]);
            const uint8_t sb = blk->scale[h];
            const __m128i c0 = _mm_load_si128((const __m128i*) rp_k16.v[sb & 15]);
            const __m128i c1 = _mm_load_si128((const __m128i*) rp_k16.v[sb >> 4]);
            const int off = 32 * h;
            for (int t = 0; t < NT; ++t) {
                const uint8_t* q8 = (const uint8_t*) y[t][i].qs + off;
                a0[t] = _mm_add_epi32(a0[t], _mm_madd_epi16(_mm_maddubs_epi16(g0, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) q8), s0)), c0));
                a1[t] = _mm_add_epi32(a1[t], _mm_madd_epi16(_mm_maddubs_epi16(g1, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + 16)), s1)), c1));
            }
        }
        const uint16_t dh = (uint16_t) (blk->d[0] | (blk->d[1] << 8));
        const float dx = h2f_local(dh) * 0.125f;
        for (int t = 0; t < NT; ++t) {
            const __m128 f = _mm_cvtepi32_ps(_mm_add_epi32(a0[t], a1[t]));
            accf[t] = _mm_add_ps(accf[t], _mm_mul_ps(_mm_set1_ps(dx * y[t][i].d), f));
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 s = _mm_add_ps(accf[t], _mm_movehl_ps(accf[t], accf[t]));
        s = _mm_add_ss(s, _mm_movehdup_ps(s));
        res[t] = _mm_cvtss_f32(s);
    }
}

// gate/up rows of one expert, repacked layout, mirroring gu_rows (gate dot, up dot, then the SwiGLU)
template <int NT>
void rp_gu_rows(const Rp22* gate, const Rp22* up, int n, const void* const* act, float* const* ff, int nrow) {
    const block_q8_K* y[NT];
    const int nb = n / QK_K;
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    for (int r = 0; r < nrow; ++r) {
        float g[NT], u[NT];
        rp_row_dot<NT>(gate + (size_t) r * nb, nb, y, g);
        rp_row_dot<NT>(up + (size_t) r * nb, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

// repack gate+up of nexp experts; returns gate/up pointers per expert
struct RpBlobs { std::vector<Rp22> buf; size_t nbrow; };
RpBlobs rp_build(const std::vector<uint8_t>& blob, size_t per, size_t gu_row, size_t up_off, int nexp, int nb, int nrow) {
    RpBlobs rb; rb.nbrow = (size_t) nrow * nb;
    rb.buf.resize((size_t) nexp * 2 * rb.nbrow);
    for (int e = 0; e < nexp; ++e) {
        const uint8_t* src = blob.data() + (size_t) e * per;
        Rp22* g = rb.buf.data() + (size_t) e * 2 * rb.nbrow;
        for (int r = 0; r < nrow; ++r)
            for (int i = 0; i < nb; ++i) {
                rp22_block(src + (size_t) r * gu_row + (size_t) i * 82, g + (size_t) r * nb + i);
                rp22_block(src + up_off + (size_t) r * gu_row + (size_t) i * 82, g + rb.nbrow + (size_t) r * nb + i);
            }
    }
    return rb;
}

void rp_gu_nt(int nt, const Rp22* g, const Rp22* u, int n, const void* const* act, float* const* ff, int nrow) {
    switch (nt) {
        case 1: rp_gu_rows<1>(g, u, n, act, ff, nrow); break;
        case 2: rp_gu_rows<2>(g, u, n, act, ff, nrow); break;
        case 3: rp_gu_rows<3>(g, u, n, act, ff, nrow); break;
        case 4: rp_gu_rows<4>(g, u, n, act, ff, nrow); break;
        case 5: rp_gu_rows<5>(g, u, n, act, ff, nrow); break;
        case 6: rp_gu_rows<6>(g, u, n, act, ff, nrow); break;
        case 7: rp_gu_rows<7>(g, u, n, act, ff, nrow); break;
        default: rp_gu_rows<8>(g, u, n, act, ff, nrow); break;
    }
}

// ---- the checks
void run_checks() {
    for (const Case& c : kCases) {
        uint64_t seed = 0x5EED1234u + (uint64_t) c.gu * 1000 + (uint64_t) c.dn;
        Fixture fx;
        make_fixture(fx, c.gu, c.dn, kMaxT, seed);
        const void* actp[kMaxT]; const void* hqp[kMaxT];
        std::vector<float> ff[kMaxT], out[kMaxT];
        float* ffp[kMaxT]; float* outp[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            actp[t] = fx.act[t].data(); hqp[t] = fx.hq[t].data();
            ff[t].resize(kFF); out[t].resize(kH);
            ffp[t] = ff[t].data(); outp[t] = out[t].data();
        }
        // gate/up: nt = 1..8, every width bit-identical to that token's nt = 1 row (#152), all rel <= 1e-5 vs ggml
        std::vector<std::vector<float>> ff1(kMaxT), out1(kMaxT);
        RpBlobs rb;   // PROTOTYPE: repacked gate/up for the IQ2_S rungs
        if (c.gu == 22) rb = rp_build(fx.blob, fx.f.bytes, fx.f.gu_row, fx.f.up_off, 1, kH / QK_K, kFF);
        for (int nt = 1; nt <= kMaxT; ++nt) {
            cpu::iq128_gu_rows(c.gu, fx.blob.data(), fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffp, 0, kFF);
            if (c.gu == 22) {
                // the bar for a repacked layout: bit for bit the rows the current kernel writes, no re-baseline
                std::vector<std::vector<float>> ffr(kMaxT, std::vector<float>(kFF));
                float* ffrp[kMaxT];
                for (int t = 0; t < kMaxT; ++t) ffrp[t] = ffr[t].data();
                rp_gu_nt(nt, rb.buf.data(), rb.buf.data() + rb.nbrow, kH, actp, ffrp, kFF);
                // exact=true above: every row compared with memcmp, so 0 failures means the repacked kernel
                // writes bit for bit what the deployed kernel writes. Say it once, where an operator reads it.
                if (nt == kMaxT) std::printf("  %-16s gu repacked: bit-identical to the deployed kernel at nt=1..%d\n",
                                             c.name, kMaxT);
                for (int t = 0; t < nt; ++t) {
                    char what[96];
                    std::snprintf(what, sizeof what, "%s gu rp vs current nt=%d token %d", c.name, nt, t);
                    check(what, ffr[t], ff[t], true);
                }
            }
            for (int t = 0; t < nt; ++t) {
                char what[96];
                std::snprintf(what, sizeof what, "%s gu nt=%d vs ggml", c.name, nt);
                check(what, ff[t], std::vector<float>(fx.ff_ref.begin() + (size_t) t * kFF,
                                                      fx.ff_ref.begin() + (size_t) (t + 1) * kFF), false);
                if (nt == 1) ff1[t] = ff[t];
                else {
                    std::snprintf(what, sizeof what, "%s gu nt=%d token %d vs its nt=1 row", c.name, nt, t);
                    check(what, ff[t], ff1[t], true);
                }
            }
        }
        // down
        for (int nt = 1; nt <= kMaxT; ++nt) {
            cpu::iq128_down_rows(c.dn, fx.blob.data() + fx.f.down_off, fx.f.d_row, kFF, hqp, nt, outp, 0, kH);
            for (int t = 0; t < nt; ++t) {
                char what[96];
                std::snprintf(what, sizeof what, "%s down nt=%d vs ggml", c.name, nt);
                check(what, out[t], std::vector<float>(fx.out_ref.begin() + (size_t) t * kH,
                                                      fx.out_ref.begin() + (size_t) (t + 1) * kH), false);
                if (nt == 1) out1[t] = out[t];
                else {
                    std::snprintf(what, sizeof what, "%s down nt=%d token %d vs its nt=1 row", c.name, nt, t);
                    check(what, out[t], out1[t], true);
                }
            }
        }
        std::printf("  %-16s checked (gu %d, down %d)\n", c.name, c.gu, c.dn);
    }
}

// ---- the bench: ms per expert, the weights streamed from DRAM
void run_bench(int mb, const std::vector<int>& nts) {
    for (const Case& c : kCases) {
        uint64_t seed = 0xBEEF0000u + (uint64_t) c.gu * 1000 + (uint64_t) c.dn;
        Fixture fx;
        make_fixture(fx, c.gu, c.dn, 1, seed);
        const size_t per = fx.f.bytes;
        const int nexp = (int) ((size_t) mb * 1024 * 1024 / per);
        std::vector<uint8_t> blob((size_t) nexp * per);
        for (int e = 0; e < nexp; ++e) std::memcpy(blob.data() + (size_t) e * per, fx.blob.data(), per);
        // vary the blocks a little so the run is not one blob repeated (the same bytes hit the same caches)
        for (int e = 1; e < nexp; ++e) {
            uint64_t s2 = seed + (uint64_t) e * 7919;
            const size_t o = (size_t) e * per;
            for (size_t i = 0; i + 8 <= per; i += 128) {
                const uint64_t v = mix(s2);
                std::memcpy(blob.data() + o + i, &v, 8);   // keep block scales (first 2 bytes of each block)
            }
        }
        for (int nt : nts) {
            if (nt > kMaxT) continue;
            const void* actp[kMaxT]; const void* hqp[kMaxT];
            float* ffp[kMaxT]; float* outp[kMaxT];
            std::vector<std::vector<float>> ff(kMaxT, std::vector<float>(kFF)), out(kMaxT, std::vector<float>(kH));
            Fixture fx8;   // nt tokens of activations
            uint64_t s8 = seed + 0x1234;
            make_fixture(fx8, c.gu, c.dn, nt, s8);
            for (int t = 0; t < nt; ++t) {
                actp[t] = fx8.act[t].data(); hqp[t] = fx8.hq[t].data();
                ffp[t] = ff[t].data(); outp[t] = out[t].data();
            }
            const ggml_vec_dot_t gu_dot = traits_cpu(c.gu)->vec_dot, dn_dot = traits_cpu(c.dn)->vec_dot;
            // ggml-cpu's path (what the engine runs today on this CPU): per-token dots over every row
            double ms_ggml_gu = 0, ms_ggml_dn = 0;
            auto t0 = Clock::now();
            for (int e = 0; e < nexp; ++e) {
                for (int r = 0; r < kFF; ++r) {
                    for (int t = 0; t < nt; ++t) {
                        float g = 0.f, u = 0.f;
                        gu_dot(kH, &g, 0, blob.data() + (size_t) e * per + (size_t) r * fx.f.gu_row, 0, (const uint8_t*) actp[t], 0, 1);
                        gu_dot(kH, &u, 0, blob.data() + (size_t) e * per + fx.f.up_off + (size_t) r * fx.f.gu_row, 0, (const uint8_t*) actp[t], 0, 1);
                        ff[t][r] = (g / (1.f + std::exp(-g))) * u;
                    }
                }
            }
            ms_ggml_gu = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            t0 = Clock::now();
            for (int e = 0; e < nexp; ++e) {
                for (int r = 0; r < kH; ++r) {
                    for (int t = 0; t < nt; ++t) {
                        float s = 0.f;
                        dn_dot(kFF, &s, 0, blob.data() + (size_t) e * per + fx.f.down_off + (size_t) r * fx.f.d_row, 0, (const uint8_t*) hqp[t], 0, 1);
                        out[t][r] = s;
                    }
                }
            }
            const double ms_ggml = ms_ggml_gu + (ms_ggml_dn = std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            double ms_iq_gu = 0, ms_iq_dn = 0;
            t0 = Clock::now();
            for (int e = 0; e < nexp; ++e) {
                cpu::iq128_gu_rows(c.gu, blob.data() + (size_t) e * per, fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffp, 0, kFF);
            }
            ms_iq_gu = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            t0 = Clock::now();
            for (int e = 0; e < nexp; ++e) {
                cpu::iq128_down_rows(c.dn, blob.data() + (size_t) e * per + fx.f.down_off, fx.f.d_row, kFF, hqp, nt, outp, 0, kH);
            }
            const double ms_iq = ms_iq_gu + (ms_iq_dn = std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            double ms_rp_gu = 0, ms_iq_gu_best = 0;
            if (c.gu == 22) {   // PROTOTYPE: same weights, grid indices pre-merged (106-byte blocks).
                // Alternate the two variants and take the best of 3 - a single shot wanders by 2x at this size.
                RpBlobs rb = rp_build(blob, per, fx.f.gu_row, fx.f.up_off, nexp, kH / QK_K, kFF);
                double best_iq = 1e30, best_rp = 1e30;
                for (int rep = 0; rep < 3; ++rep) {
                    t0 = Clock::now();
                    for (int e = 0; e < nexp; ++e)
                        cpu::iq128_gu_rows(c.gu, blob.data() + (size_t) e * per, fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffp, 0, kFF);
                    best_iq = std::min(best_iq, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                    t0 = Clock::now();
                    for (int e = 0; e < nexp; ++e) {
                        const Rp22* g = rb.buf.data() + (size_t) e * 2 * rb.nbrow;
                        rp_gu_nt(nt, g, g + rb.nbrow, kH, actp, ffp, kFF);
                    }
                    best_rp = std::min(best_rp, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                }
                ms_iq_gu_best = best_iq / nexp; ms_rp_gu = best_rp / nexp;
            }
            std::printf("  %-16s nt=%d  ggml %8.2f ms/expert   iq128 %6.2f ms/expert   %.1fx   [gu %.2f vs %.2f = %.2fx  dn %.2f vs %.2f = %.2fx]\n",
                        c.name, nt, ms_ggml / nexp, ms_iq / nexp, ms_ggml / ms_iq,
                        ms_ggml_gu / nexp, ms_iq_gu / nexp, ms_ggml_gu / ms_iq_gu,
                        ms_ggml_dn / nexp, ms_iq_dn / nexp, ms_ggml_dn / ms_iq_dn);
            if (ms_rp_gu > 0)
                std::printf("  %-16s nt=%d  gu  current %.3f  vs  REPACKED %.3f ms/expert  = %.3fx (best of 3, alternating)\n",
                            c.name, nt, ms_iq_gu_best, ms_rp_gu, ms_iq_gu_best / ms_rp_gu);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int mb = 128, cpu_pin = -1;
    std::vector<int> nts = {1, 2, 4};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench") bench = true;
        else if (a == "--mb" && i + 1 < argc) mb = std::atoi(argv[++i]);
        else if (a == "--cpu" && i + 1 < argc) cpu_pin = std::atoi(argv[++i]);
        else if (a == "--nt" && i + 1 < argc) {
            nts.clear();
            const char* p = argv[++i];
            while (*p) { nts.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p == ',') ++p; }
        } else {
            std::printf("usage: iq_avx1_parity [--bench [--mb N] [--nt 1,2,4] [--cpu N]]\n");
            return 2;
        }
    }
#if !defined(_WIN32)
    if (cpu_pin >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu_pin, &set);
        if (sched_setaffinity(0, sizeof set, &set) != 0) std::fprintf(stderr, "pinning to CPU %d failed\n", cpu_pin);
    }
#endif
    // The IQ3_XXS/IQ3_S fold adds two int16 partial dots before the scale multiply, so 4*127*grid_max has to
    // stay inside int16.  Scan the tables: they pack four magnitudes per uint32 entry in this ggml.
    auto gmax = [](const uint32_t* tab, int n) { int m = 0;
        for (int i = 0; i < n; ++i) for (int s = 0; s < 4; ++s) m = std::max(m, (int) ((tab[i] >> (8 * s)) & 0xFF));
        return m; };
    for (auto pr : { std::pair{"iq3xxs_grid", gmax(iq3xxs_grid, 256)}, std::pair{"iq3s_grid", gmax(iq3s_grid, 512)} })
        if (4 * 127 * pr.second >= 32768) {
            std::printf("  FAIL %s: max magnitude %d, the folded halves would saturate int16\n", pr.first, pr.second);
            ++failures;
        }
    if (!cpu::cpu_avx1_ok()) { std::printf("iq_avx1_parity: no AVX here, the kernels would not run\n"); return 0; }
    std::printf("iq_avx1_parity: %s\n", cpu::cpu_avx1_ok() ? "AVX1 kernels checked (128-bit lanes)" : "");
    if (bench) { run_bench(mb, nts); return failures != 0; }
    run_checks();
    for (const auto& kv : vs_ggml) std::printf("iq_avx1_parity: vs ggml-cpu %-24s %6llu rows differ bitwise, worst rel %.3e\n", kv.first.c_str(), (unsigned long long) kv.second.first, kv.second.second);
    for (const auto& kv : rung_fp) std::printf("iq_avx1_parity: rung fingerprint %-22s %016llx\n", kv.first.c_str(), (unsigned long long) kv.second);
    std::printf("iq_avx1_parity: row fingerprint %016llx\n", (unsigned long long) row_fp);
    std::printf(failures ? "iq_avx1_parity: %d FAILURES\n" : "iq_avx1_parity: all rows match\n", failures);
    return failures != 0;
}
