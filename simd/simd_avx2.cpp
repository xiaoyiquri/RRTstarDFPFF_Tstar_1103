#include "simd/simd_backend.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <vector>

namespace rrtpff::avx2 {
// Deliberately keep multiply/add separate: root and boundary decisions use the
// scalar operation order. No fast-math and no float reduction of planner state.
void binary64(const double *a, const double *b, double *out, std::size_t n, detail::BinaryOp op) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        auto x = _mm256_loadu_pd(a + i), y = _mm256_loadu_pd(b + i);
        __m256d z;
        switch (op) {
        case detail::BinaryOp::Add:
            z = _mm256_add_pd(x, y);
            break;
        case detail::BinaryOp::Subtract:
            z = _mm256_sub_pd(x, y);
            break;
        case detail::BinaryOp::Multiply:
            z = _mm256_mul_pd(x, y);
            break;
        case detail::BinaryOp::Divide:
            z = _mm256_div_pd(x, y);
            break;
        case detail::BinaryOp::Minimum:
            z = _mm256_min_pd(x, y);
            break;
        default:
            z = _mm256_max_pd(x, y);
            break;
        }
        _mm256_storeu_pd(out + i, z);
    }
    detail::binary64(a + i, b + i, out + i, n - i, op, SIMDBackend::Scalar);
}
void binary32(const float *a, const float *b, float *out, std::size_t n, detail::BinaryOp op) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        auto x = _mm256_loadu_ps(a + i), y = _mm256_loadu_ps(b + i);
        __m256 z;
        switch (op) {
        case detail::BinaryOp::Add:
            z = _mm256_add_ps(x, y);
            break;
        case detail::BinaryOp::Subtract:
            z = _mm256_sub_ps(x, y);
            break;
        case detail::BinaryOp::Multiply:
            z = _mm256_mul_ps(x, y);
            break;
        case detail::BinaryOp::Divide:
            z = _mm256_div_ps(x, y);
            break;
        case detail::BinaryOp::Minimum:
            z = _mm256_min_ps(x, y);
            break;
        default:
            z = _mm256_max_ps(x, y);
            break;
        }
        _mm256_storeu_ps(out + i, z);
    }
    detail::binary32(a + i, b + i, out + i, n - i, op, SIMDBackend::Scalar);
}
void polynomial(const double *c, int degree, const double *t, std::size_t n, double *out) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        auto time = _mm256_loadu_pd(t + i), v = _mm256_setzero_pd();
        for (int k = degree; k >= 0; --k)
            v = _mm256_add_pd(_mm256_mul_pd(v, time), _mm256_loadu_pd(c + k * n + i));
        _mm256_storeu_pd(out + i, v);
    }
    for (; i < n; ++i) {
        double v = 0;
        for (int k = degree; k >= 0; --k)
            v = v * t[i] + c[k * n + i];
        out[i] = v;
    }
}
void polynomial32(const float *c, int degree, const float *t, std::size_t n, float *out) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        auto time = _mm256_loadu_ps(t + i);
        auto value = _mm256_setzero_ps();
        for (int k = degree; k >= 0; --k)
            value = _mm256_add_ps(_mm256_mul_ps(value, time), _mm256_loadu_ps(c + k * n + i));
        _mm256_storeu_ps(out + i, value);
    }
    for (; i < n; ++i) {
        float value = 0;
        for (int k = degree; k >= 0; --k)
            value = value * t[i] + c[k * n + i];
        out[i] = value;
    }
}
void distance(const double *x, const double *q, const double *w, const double *periods, int dim,
              std::size_t n, std::size_t stride, double *out) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        auto sum = _mm256_setzero_pd();
        for (int a = 0; a < dim; ++a) {
            auto d = _mm256_sub_pd(_mm256_loadu_pd(x + a * stride + i), _mm256_set1_pd(q[a]));
            if (periods[a] > 0) {
                auto period = _mm256_set1_pd(periods[a]);
                auto turns =
                    _mm256_floor_pd(_mm256_add_pd(_mm256_div_pd(d, period), _mm256_set1_pd(.5)));
                d = _mm256_sub_pd(d, _mm256_mul_pd(period, turns));
            }
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(w[a]), d), d));
        }
        _mm256_storeu_pd(out + i, sum);
    }
    for (; i < n; ++i) {
        double sum = 0;
        for (int a = 0; a < dim; ++a) {
            double d = x[a * stride + i] - q[a];
            if (periods[a] > 0)
                d -= periods[a] * std::floor(d / periods[a] + .5);
            sum += w[a] * d * d;
        }
        out[i] = sum;
    }
}
void energy(const double *basis, const double *inv, int rows, int order, std::size_t n,
            double *out) {
    std::vector<double> solved(rows * order * n, 0.0);
    for (int l = 0; l < order; ++l)
        for (int r = 0; r < rows; ++r)
            for (int s = 0; s < rows; ++s) {
                const double weight = inv[r * rows + s];
                if (weight == 0.0)
                    continue;
                auto scale = _mm256_set1_pd(weight);
                auto *dst = solved.data() + (r * order + l) * n;
                const auto *src = basis + (s * order + l) * n;
                std::size_t i = 0;
                for (; i + 4 <= n; i += 4)
                    _mm256_storeu_pd(dst + i,
                                     _mm256_add_pd(_mm256_loadu_pd(dst + i),
                                                   _mm256_mul_pd(scale, _mm256_loadu_pd(src + i))));
                for (; i < n; ++i)
                    dst[i] += weight * src[i];
            }
    std::fill(out, out + (2 * order - 1) * n, 0.0);
    for (int k = 0; k < order; ++k)
        for (int l = 0; l < order; ++l)
            for (int r = 0; r < rows; ++r) {
                auto *dst = out + (k + l) * n;
                const auto *a = basis + (r * order + k) * n;
                const auto *b = solved.data() + (r * order + l) * n;
                std::size_t i = 0;
                for (; i + 4 <= n; i += 4)
                    _mm256_storeu_pd(dst + i, _mm256_add_pd(_mm256_loadu_pd(dst + i),
                                                            _mm256_mul_pd(_mm256_loadu_pd(a + i),
                                                                          _mm256_loadu_pd(b + i))));
                for (; i < n; ++i)
                    dst[i] += a[i] * b[i];
            }
}
void sphere(const double *x, const double *center, int dim, double radius, std::size_t n,
            unsigned char *hit) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        auto sum = _mm256_setzero_pd();
        for (int a = 0; a < dim; ++a) {
            auto d = _mm256_sub_pd(_mm256_loadu_pd(x + a * n + i), _mm256_set1_pd(center[a]));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(d, d));
        }
        int mask =
            _mm256_movemask_pd(_mm256_cmp_pd(sum, _mm256_set1_pd(radius * radius), _CMP_LE_OQ));
        for (int lane = 0; lane < 4; ++lane)
            hit[i + lane] = static_cast<unsigned char>((mask >> lane) & 1);
    }
    for (; i < n; ++i) {
        double sum = 0;
        for (int a = 0; a < dim; ++a) {
            double d = x[a * n + i] - center[a];
            sum += d * d;
        }
        hit[i] = static_cast<unsigned char>(sum <= radius * radius);
    }
}
void sphereAABB(const double *x, const double *lower, const double *upper, int dim, double radius,
                std::size_t n, unsigned char *hit) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        auto sum = _mm256_setzero_pd();
        auto valid = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
        for (int a = 0; a < dim; ++a) {
            auto value = _mm256_loadu_pd(x + a * n + i);
            valid = _mm256_and_pd(valid, _mm256_cmp_pd(value, value, _CMP_ORD_Q));
            auto d = _mm256_max_pd(_mm256_setzero_pd(),
                                   _mm256_max_pd(_mm256_sub_pd(_mm256_set1_pd(lower[a]), value),
                                                 _mm256_sub_pd(value, _mm256_set1_pd(upper[a]))));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(d, d));
        }
        const int mask =
            radius > 0
                ? _mm256_movemask_pd(_mm256_and_pd(
                      valid, _mm256_cmp_pd(sum, _mm256_set1_pd(radius * radius), _CMP_LT_OQ)))
                : 0;
        for (int lane = 0; lane < 4; ++lane)
            hit[i + lane] = static_cast<unsigned char>((mask >> lane) & 1);
    }
    for (; i < n; ++i) {
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
void sphereAABB32(const float *x, const float *lower, const float *upper, int dim, float radius,
                  std::size_t n, unsigned char *hit) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        auto sum = _mm256_setzero_ps();
        auto valid = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
        for (int a = 0; a < dim; ++a) {
            auto value = _mm256_loadu_ps(x + a * n + i);
            valid = _mm256_and_ps(valid, _mm256_cmp_ps(value, value, _CMP_ORD_Q));
            auto d = _mm256_max_ps(_mm256_setzero_ps(),
                                   _mm256_max_ps(_mm256_sub_ps(_mm256_set1_ps(lower[a]), value),
                                                 _mm256_sub_ps(value, _mm256_set1_ps(upper[a]))));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(d, d));
        }
        const int mask = radius > 0
                             ? _mm256_movemask_ps(_mm256_and_ps(
                                   valid, _mm256_cmp_ps(sum, _mm256_set1_ps(radius * radius),
                                                        _CMP_LT_OQ)))
                             : 0;
        for (int lane = 0; lane < 8; ++lane)
            hit[i + lane] = static_cast<unsigned char>((mask >> lane) & 1);
    }
    for (; i < n; ++i) {
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
} // namespace rrtpff::avx2
