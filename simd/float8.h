#pragma once
// Included only by the AVX2 translation unit. No scalar transcendental fallback.
#include <immintrin.h>
namespace rrtpff::wide {
struct F {
    __m256 v;
    F() : v(_mm256_setzero_ps()) {}
    F(float x) : v(_mm256_set1_ps(x)) {}
    explicit F(__m256 x) : v(x) {}
    static F load(const float *p) { return F(_mm256_loadu_ps(p)); }
    void store(float *p) const { _mm256_storeu_ps(p, v); }
};
inline F operator+(F a, F b) { return F(_mm256_add_ps(a.v, b.v)); }
inline F operator-(F a, F b) { return F(_mm256_sub_ps(a.v, b.v)); }
inline F operator-(F a) { return F(0) - a; }
inline F operator*(F a, F b) { return F(_mm256_mul_ps(a.v, b.v)); }
inline F operator/(F a, F b) { return F(_mm256_div_ps(a.v, b.v)); }
inline F operator&(F a, F b) { return F(_mm256_and_ps(a.v, b.v)); }
inline F operator|(F a, F b) { return F(_mm256_or_ps(a.v, b.v)); }
inline F operator>(F a, F b) { return F(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ)); }
inline F operator<(F a, F b) { return F(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ)); }
inline F operator<=(F a, F b) { return F(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ)); }
inline F select(F mask, F yes, F no) { return F(_mm256_blendv_ps(no.v, yes.v, mask.v)); }
inline F abs(F a) { return F(_mm256_andnot_ps(_mm256_set1_ps(-0.f), a.v)); }
inline F sqrt(F a) { return F(_mm256_sqrt_ps(a.v)); }
inline F max(F a, F b) { return F(_mm256_max_ps(a.v, b.v)); }
inline F min(F a, F b) { return F(_mm256_min_ps(a.v, b.v)); }
inline F finite(F a) { return abs(a) <= F(3.402823466e38f); }
inline unsigned bits(F a) { return static_cast<unsigned>(_mm256_movemask_ps(a.v)); }
inline F sign(F magnitude, F sign_source) {
    return F(_mm256_or_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), magnitude.v),
                          _mm256_and_ps(_mm256_set1_ps(-0.f), sign_source.v)));
}
// Reduce to [-pi,pi]; inputs outside the documented range are rejected by DF.
inline F wrap(F a) {
    F n(_mm256_round_ps((a / F(6.283185307179586f)).v,
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return a - n * F(6.283185307179586f);
}
inline void sincos(F angle, F &s, F &c) {
    F x = wrap(angle), ax = abs(x);
    F q = select(ax > F(1.570796326794897f), sign(F(3.141592653589793f), x) - x, x);
    F z = q * q;
    s = q * (F(1) + z * (F(-1.f / 6) +
                         z * (F(1.f / 120) + z * (F(-1.f / 5040) + z * (F(1.f / 362880) +
                                                                        z * F(-1.f / 39916800))))));
    F cc = F(1) +
           z * (F(-.5f) +
                z * (F(1.f / 24) +
                     z * (F(-1.f / 720) + z * (F(1.f / 40320) +
                                               z * (F(-1.f / 3628800) + z * F(1.f / 479001600))))));
    c = select(ax > F(1.570796326794897f), -cc, cc);
}
// atan uses a pi/4 reduction, then the alternating series on |z|<=tan(pi/8).
inline F atan2(F y, F x) {
    F ax = abs(x), ay = abs(y);
    F t = min(ax, ay) / max(max(ax, ay), F(1e-30f));
    F reduce = t > F(.414213562373095f);
    F z = select(reduce, (t - F(1)) / (t + F(1)), t), zz = z * z;
    F a = z *
          (F(1) +
           zz * (F(-1.f / 3) +
                 zz * (F(1.f / 5) +
                       zz * (F(-1.f / 7) +
                             zz * (F(1.f / 9) + zz * (F(-1.f / 11) +
                                                      zz * (F(1.f / 13) + zz * F(-1.f / 15))))))));
    a = a + select(reduce, F(.785398163397448f), F(0));
    a = select(ay > ax, F(1.570796326794897f) - a, a);
    a = select(x < F(0), F(3.141592653589793f) - a, a);
    return sign(a, y);
}
struct V3 {
    F x, y, z;
    F &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    F operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, F b) { return {a.x * b, a.y * b, a.z * b}; }
inline V3 operator/(V3 a, F b) { return {a.x / b, a.y / b, a.z / b}; }
inline F dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
} // namespace rrtpff::wide
