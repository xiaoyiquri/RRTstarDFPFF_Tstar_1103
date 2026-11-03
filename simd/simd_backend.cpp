#include "simd/simd_backend.h"
#include <algorithm>
#include <cmath>
#include <vector>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace rrtpff {
#ifdef RRTPFF_HAS_AVX2
namespace avx2 {
void binary64(const double *, const double *, double *, std::size_t, detail::BinaryOp);
void binary32(const float *, const float *, float *, std::size_t, detail::BinaryOp);
void polynomial(const double *, int, const double *, std::size_t, double *);
void polynomial32(const float *, int, const float *, std::size_t, float *);
void distance(const double *, const double *, const double *, const double *, int, std::size_t,
              std::size_t, double *);
void energy(const double *, const double *, int, int, std::size_t, double *);
void sphere(const double *, const double *, int, double, std::size_t, unsigned char *);
void sphereAABB(const double *, const double *, const double *, int, double, std::size_t,
                unsigned char *);
void sphereAABB32(const float *, const float *, const float *, int, float, std::size_t,
                  unsigned char *);
} // namespace avx2
#endif
bool simdCompiledAVX2() {
#ifdef RRTPFF_HAS_AVX2
    return true;
#else
    return false;
#endif
}
bool simdRuntimeAVX2() {
    static const bool supported = [] {
#ifndef RRTPFF_HAS_AVX2
        return false;
#elif defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        if (info[0] < 7)
            return false;
        __cpuid(info, 1);
        if ((info[2] & (1 << 27)) == 0 || (info[2] & (1 << 28)) == 0)
            return false;
        if ((_xgetbv(0) & 6) != 6)
            return false;
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
        // Direct CPUID avoids requiring GCC's __cpu_model runtime on Clang/
        // compiler-rt Windows toolchains. Check OS YMM state before AVX2.
        if (__get_cpuid_max(0, nullptr) < 7)
            return false;
        unsigned int eax, ebx, ecx, edx;
        __cpuid_count(1, 0, eax, ebx, ecx, edx);
        if ((ecx & (1u << 27)) == 0 || (ecx & (1u << 28)) == 0)
            return false;
        unsigned int xcr0_low, xcr0_high;
        __asm__ volatile("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0));
        if ((xcr0_low & 6u) != 6u)
            return false;
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1u << 5)) != 0;
#else
        return false;
#endif
    }();
    return supported;
}
SIMDBackend selectSIMDBackend(bool enabled) {
    return enabled && simdRuntimeAVX2() ? SIMDBackend::AVX2 : SIMDBackend::Scalar;
}
namespace detail {
template <class T> void scalarBinary(const T *a, const T *b, T *out, std::size_t n, BinaryOp op) {
    for (std::size_t i = 0; i < n; ++i) {
        switch (op) {
        case BinaryOp::Add:
            out[i] = a[i] + b[i];
            break;
        case BinaryOp::Subtract:
            out[i] = a[i] - b[i];
            break;
        case BinaryOp::Multiply:
            out[i] = a[i] * b[i];
            break;
        case BinaryOp::Divide:
            out[i] = a[i] / b[i];
            break;
        case BinaryOp::Minimum:
            out[i] = std::fmin(a[i], b[i]);
            break;
        case BinaryOp::Maximum:
            out[i] = std::fmax(a[i], b[i]);
            break;
        }
    }
}
void binary64(const double *a, const double *b, double *out, std::size_t n, BinaryOp op,
              SIMDBackend backend) {
#ifdef RRTPFF_HAS_AVX2
    if (backend == SIMDBackend::AVX2 && simdRuntimeAVX2())
        return avx2::binary64(a, b, out, n, op);
#endif
    (void)backend;
    scalarBinary(a, b, out, n, op);
}
void binary32(const float *a, const float *b, float *out, std::size_t n, BinaryOp op,
              SIMDBackend backend) {
#ifdef RRTPFF_HAS_AVX2
    if (backend == SIMDBackend::AVX2 && simdRuntimeAVX2())
        return avx2::binary32(a, b, out, n, op);
#endif
    (void)backend;
    scalarBinary(a, b, out, n, op);
}
} // namespace detail
void polynomialBatch(const double *c, int degree, const double *t, std::size_t n, double *out,
                     SIMDBackend b) {
#ifdef RRTPFF_HAS_AVX2
    if (b == SIMDBackend::AVX2 && simdRuntimeAVX2())
        return avx2::polynomial(c, degree, t, n, out);
#endif
    (void)b;
    for (std::size_t i = 0; i < n; ++i) {
        double v = 0;
        for (int k = degree; k >= 0; --k)
            v = v * t[i] + c[k * n + i];
        out[i] = v;
    }
}
void polynomialBatchFloat(const float *c, int degree, const float *t, std::size_t n, float *out,
                          SIMDBackend b) {
#ifdef RRTPFF_HAS_AVX2
    if (b == SIMDBackend::AVX2 && simdRuntimeAVX2())
        return avx2::polynomial32(c, degree, t, n, out);
#endif
    (void)b;
    for (std::size_t i = 0; i < n; ++i) {
        float v = 0;
        for (int k = degree; k >= 0; --k)
            v = v * t[i] + c[k * n + i];
        out[i] = v;
    }
}
void distanceBatch(const double *x, const double *q, const double *w, const double *periods,
                   int dim, std::size_t n, std::size_t stride, double *out, SIMDBackend b) {
#ifdef RRTPFF_HAS_AVX2
    if (b == SIMDBackend::AVX2 && simdRuntimeAVX2())
        return avx2::distance(x, q, w, periods, dim, n, stride, out);
#endif
    (void)b;
    std::fill(out, out + n, 0.0);
    for (int a = 0; a < dim; ++a)
        for (std::size_t i = 0; i < n; ++i) {
            double d = x[a * stride + i] - q[a];
            if (periods[a] > 0)
                d -= periods[a] * std::floor(d / periods[a] + .5);
            out[i] += w[a] * d * d;
        }
}
void energyBatch(const double *basis, const double *inv, int rows, int order, std::size_t n,
                 double *out, SIMDBackend b) {
#ifdef RRTPFF_HAS_AVX2
    if (b == SIMDBackend::AVX2 && simdRuntimeAVX2())
        return avx2::energy(basis, inv, rows, order, n, out);
#endif
    (void)b;
    // First compute inverse * basis, then b_k^T * solved_l. This avoids
    // recomputing the same matrix product for every pair of polynomial powers.
    std::vector<double> solved(rows * order * n, 0.0);
    for (int l = 0; l < order; ++l)
        for (int r = 0; r < rows; ++r)
            for (int s = 0; s < rows; ++s)
                for (std::size_t i = 0; i < n; ++i)
                    solved[(r * order + l) * n + i] +=
                        inv[r * rows + s] * basis[(s * order + l) * n + i];
    std::fill(out, out + (2 * order - 1) * n, 0.0);
    for (int k = 0; k < order; ++k)
        for (int l = 0; l < order; ++l)
            for (int r = 0; r < rows; ++r)
                for (std::size_t i = 0; i < n; ++i)
                    out[(k + l) * n + i] +=
                        basis[(r * order + k) * n + i] * solved[(r * order + l) * n + i];
}
void sphereBatch(const double *x, const double *center, int dim, double radius, std::size_t n,
                 unsigned char *hit, SIMDBackend b) {
#ifdef RRTPFF_HAS_AVX2
    if (b == SIMDBackend::AVX2 && simdRuntimeAVX2())
        return avx2::sphere(x, center, dim, radius, n, hit);
#endif
    (void)b;
    for (std::size_t i = 0; i < n; ++i) {
        double d2 = 0;
        for (int a = 0; a < dim; ++a) {
            double d = x[a * n + i] - center[a];
            d2 += d * d;
        }
        hit[i] = static_cast<unsigned char>(d2 <= radius * radius);
    }
}
void sphereAABBBatch(const double *x, const double *lower, const double *upper, int dim,
                     double radius, std::size_t n, unsigned char *hit, SIMDBackend b) {
#ifdef RRTPFF_HAS_AVX2
    if (b == SIMDBackend::AVX2 && simdRuntimeAVX2())
        return avx2::sphereAABB(x, lower, upper, dim, radius, n, hit);
#endif
    (void)b;
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0;
        bool finite = true;
        for (int a = 0; a < dim; ++a) {
            const double value = x[a * n + i];
            finite = finite && std::isfinite(value);
            const double d = std::max({lower[a] - value, value - upper[a], 0.0});
            sum += d * d;
        }
        hit[i] = static_cast<unsigned char>(finite && radius > 0 && sum < radius * radius);
    }
}
void sphereAABBBatchFloat(const float *x, const float *lower, const float *upper, int dim,
                          float radius, std::size_t n, unsigned char *hit, SIMDBackend b) {
#ifdef RRTPFF_HAS_AVX2
    if (b == SIMDBackend::AVX2 && simdRuntimeAVX2())
        return avx2::sphereAABB32(x, lower, upper, dim, radius, n, hit);
#endif
    (void)b;
    for (std::size_t i = 0; i < n; ++i) {
        float sum = 0;
        bool finite = true;
        for (int a = 0; a < dim; ++a) {
            const float value = x[a * n + i];
            finite = finite && std::isfinite(value);
            const float d = std::max({lower[a] - value, value - upper[a], 0.0f});
            sum += d * d;
        }
        hit[i] = static_cast<unsigned char>(finite && radius > 0 && sum < radius * radius);
    }
}
} // namespace rrtpff
