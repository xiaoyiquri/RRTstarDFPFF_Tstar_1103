#pragma once
#include "common/math_utils.h"
#include "simd/simd_backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace rrtpff {
// Derivative isolation preserves all monotone brackets and even-multiplicity
// roots. Only independent bracket refinement uses SIMD; small residuals are
// evaluated with the scalar long-double oracle before a sign decision.
template <std::size_t Width>
std::array<std::vector<double>, Width>
batchRoots(std::array<std::vector<long double>, Width> polynomials, double lo, double hi,
           std::size_t count, SIMDBackend backend) {
    std::array<std::vector<double>, Width> roots{};
    if (count > Width || !(hi >= lo))
        return roots;
    std::array<std::vector<long double>, Width> derivatives{};
    bool recurse = false;
    for (std::size_t i = 0; i < count; ++i) {
        auto &c = polynomials[i];
        while (c.size() > 1 && c.back() == 0)
            c.pop_back();
        if (c.size() == 2) {
            double x = static_cast<double>(-c[0] / c[1]);
            if (std::isfinite(x) && x >= lo && x <= hi)
                roots[i].push_back(x);
        } else if (c.size() > 2) {
            recurse = true;
            derivatives[i].resize(c.size() - 1);
            for (std::size_t k = 1; k < c.size(); ++k)
                derivatives[i][k - 1] = k * c[k];
        }
    }
    if (!recurse)
        return roots;
    auto critical = batchRoots<Width>(derivatives, lo, hi, count, backend);
    struct Bracket {
        std::size_t lane;
        double a, b;
        long double fa;
    };
    std::vector<Bracket> brackets;
    for (std::size_t i = 0; i < count; ++i) {
        const auto &c = polynomials[i];
        if (c.size() < 3)
            continue;
        auto cuts = std::move(critical[i]);
        cuts.insert(cuts.begin(), lo);
        cuts.push_back(hi);
        for (double t : cuts)
            if (std::abs(polynomialValue(c, t)) <=
                1e-13L * std::max(1e-300L, polynomialScale(c, t)))
                roots[i].push_back(t);
        for (std::size_t j = 1; j < cuts.size(); ++j) {
            auto fa = polynomialValue(c, cuts[j - 1]), fb = polynomialValue(c, cuts[j]);
            if (fa != 0 && fb != 0 && ((fa < 0) != (fb < 0)))
                brackets.push_back({i, cuts[j - 1], cuts[j], fa});
        }
    }
    for (std::size_t offset = 0; offset < brackets.size(); offset += Width) {
        const auto n = std::min(Width, brackets.size() - offset);
        int degree = 0;
        for (std::size_t i = 0; i < n; ++i)
            degree = std::max(degree,
                              static_cast<int>(polynomials[brackets[offset + i].lane].size() - 1));
        std::vector<double> coefficients((degree + 1) * n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const auto &c = polynomials[brackets[offset + i].lane];
            for (std::size_t k = 0; k < c.size(); ++k)
                coefficients[k * n + i] = static_cast<double>(c[k]);
        }
        std::array<bool, Width> done{};
        std::array<double, Width> t{}, value{};
        for (int iteration = 0; iteration < 100; ++iteration) {
            bool active = false;
            for (std::size_t i = 0; i < n; ++i) {
                auto &b = brackets[offset + i];
                t[i] = b.a + (b.b - b.a) * .5;
                active |= !done[i];
            }
            if (!active)
                break;
            polynomialBatch(coefficients.data(), degree, t.data(), n, value.data(), backend);
            for (std::size_t i = 0; i < n; ++i) {
                if (done[i])
                    continue;
                auto &b = brackets[offset + i];
                const auto &c = polynomials[b.lane];
                long double f = value[i];
                if (!std::isfinite(value[i]) || std::abs(f) <= 1e-12L * polynomialScale(c, t[i]))
                    f = polynomialValue(c, t[i]);
                if (f == 0) {
                    b.a = b.b = t[i];
                    done[i] = true;
                    continue;
                }
                if ((b.fa < 0) != (f < 0))
                    b.b = t[i];
                else {
                    b.a = t[i];
                    b.fa = f;
                }
                done[i] = b.b - b.a < 1e-12 * std::max(1.0, std::abs(t[i]));
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            const auto &b = brackets[offset + i];
            roots[b.lane].push_back(b.a + (b.b - b.a) * .5);
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        auto &r = roots[i];
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end(),
                            [](double a, double b) {
                                return std::abs(a - b) <=
                                       1e-10 * std::max({1.0, std::abs(a), std::abs(b)});
                            }),
                r.end());
    }
    return roots;
}
} // namespace rrtpff
