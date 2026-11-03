#pragma once
#include <array>
#include <cstddef>

namespace rrtpff {
inline constexpr std::size_t SIMDBatchWidth = 8;
enum class SIMDBackend { Scalar, AVX2 };
bool simdCompiledAVX2();
bool simdRuntimeAVX2();
SIMDBackend selectSIMDBackend(bool enabled);

namespace detail {
enum class BinaryOp { Add, Subtract, Multiply, Divide, Minimum, Maximum };
void binary64(const double *, const double *, double *, std::size_t, BinaryOp, SIMDBackend);
void binary32(const float *, const float *, float *, std::size_t, BinaryOp, SIMDBackend);
} // namespace detail

// No ISA-dependent layout crosses a translation-unit boundary. The AVX2
// instructions live in simd_avx2.cpp; all other code is baseline-CPU safe.
template <class T, std::size_t Width> struct SIMDVector {
    std::array<T, Width> lane{};
    SIMDBackend backend{SIMDBackend::Scalar};
    static SIMDVector load(const T *p, SIMDBackend b = selectSIMDBackend(true)) {
        SIMDVector out;
        out.backend = b;
        for (std::size_t i = 0; i < Width; ++i)
            out.lane[i] = p[i];
        return out;
    }
    void store(T *p) const {
        for (std::size_t i = 0; i < Width; ++i)
            p[i] = lane[i];
    }
    SIMDVector binary(const SIMDVector &b, detail::BinaryOp op) const;
    SIMDVector operator+(const SIMDVector &b) const { return binary(b, detail::BinaryOp::Add); }
    SIMDVector operator-(const SIMDVector &b) const {
        return binary(b, detail::BinaryOp::Subtract);
    }
    SIMDVector operator*(const SIMDVector &b) const {
        return binary(b, detail::BinaryOp::Multiply);
    }
    SIMDVector operator/(const SIMDVector &b) const { return binary(b, detail::BinaryOp::Divide); }
};
using SimdFloat8 = SIMDVector<float, 8>;
using SimdDouble4 = SIMDVector<double, 4>;
using SimdDouble8 = SIMDVector<double, 8>;

// Coefficient-major SoA: coefficients[k * count + lane], ascending k.
void polynomialBatch(const double *coefficients, int degree, const double *times, std::size_t count,
                     double *output, SIMDBackend backend);
// Paper-style float8 path: coefficient-major SoA, with eight AVX2 float lanes.
// The scalar fallback is kept for hosts without AVX2 and for reference tests.
void polynomialBatchFloat(const float *coefficients, int degree, const float *times,
                          std::size_t count, float *output, SIMDBackend backend);
// Coordinates [axis * count + node]; weights are nonnegative. Period 0 is
// Euclidean, a positive period supplies wrapped angular distance.
void distanceBatch(const double *coordinates, const double *query, const double *weights,
                   const double *periods, int dim, std::size_t count, std::size_t stride,
                   double *squared, SIMDBackend backend);
// Basis [row][order][count], inverse of normalized projected Gramian [row][row].
void energyBatch(const double *basis, const double *inverse, int rows, int order, std::size_t count,
                 double *coefficients, SIMDBackend backend);
void sphereBatch(const double *coordinates, const double *center, int dim, double radius,
                 std::size_t count, unsigned char *hit, SIMDBackend backend);
// Centers are coordinate-major SoA. Strict interior test against an AABB:
// an INSCRIBED robot sphere hit proves collision; a miss proves nothing.
void sphereAABBBatch(const double *coordinates, const double *lower, const double *upper, int dim,
                     double inner_radius, std::size_t count, unsigned char *hit,
                     SIMDBackend backend);
void sphereAABBBatchFloat(const float *coordinates, const float *lower, const float *upper,
                          int dim, float inner_radius, std::size_t count, unsigned char *hit,
                          SIMDBackend backend);

void distanceBatchFloat(const float*, const float*, const float*, const float*, int,
                        std::size_t, std::size_t, float*, SIMDBackend);
inline void distanceBatchTyped(const float* a,const float* b,const float* c,const float* d,int n,
                               std::size_t count,std::size_t stride,float* out,SIMDBackend backend) {
    distanceBatchFloat(a,b,c,d,n,count,stride,out,backend);
}
inline void distanceBatchTyped(const double* a,const double* b,const double* c,const double* d,int n,
                               std::size_t count,std::size_t stride,double* out,SIMDBackend backend) {
    distanceBatch(a,b,c,d,n,count,stride,out,backend);
}

} // namespace rrtpff

#include <type_traits>
namespace rrtpff {
template <class T, std::size_t W>
SIMDVector<T, W> SIMDVector<T, W>::binary(const SIMDVector &b, detail::BinaryOp op) const {
    SIMDVector out;
    out.backend = backend;
    if constexpr (std::is_same_v<T, double>)
        detail::binary64(lane.data(), b.lane.data(), out.lane.data(), W, op, backend);
    else
        detail::binary32(lane.data(), b.lane.data(), out.lane.data(), W, op, backend);
    return out;
}
} // namespace rrtpff
