#pragma once
#include "common/flat_trajectory.h"
#include "simd/simd_backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <variant>

namespace rrtpff {
template <int Dim, int Order, std::size_t Width = 8> struct FlatSamplesSoA {
    // [derivative][axis][lane], including derivative Order (the flat control).
    std::array<std::array<std::array<double, Width>, Dim>, Order + 1> values{};
    std::size_t count{0};
    FlatState<Dim, Order> state(std::size_t lane) const {
        if (lane >= count)
            throw std::out_of_range("Inactive sample lane");
        FlatState<Dim, Order> out;
        for (int d = 0; d < Order; ++d)
            for (int a = 0; a < Dim; ++a)
                out.values(d * Dim + a) = values[d][a][lane];
        return out;
    }
    FlatControl<Dim> control(std::size_t lane) const {
        if (lane >= count)
            throw std::out_of_range("Inactive sample lane");
        FlatControl<Dim> out;
        for (int a = 0; a < Dim; ++a)
            out.values(a) = values[Order][a][lane];
        return out;
    }
};

template <int Dim, int Order, typename Metadata = std::monostate, std::size_t Width = 8>
struct SIMDFlatEvaluator {
    using Trajectory = FlatTrajectory<Dim, Order, Metadata>;
    using State = FlatState<Dim, Order>;
    using Samples = FlatSamplesSoA<Dim, Order, Width>;

    Samples evaluateSoA(const std::array<const Trajectory *, Width> &trajectories,
                        const std::array<double, Width> &times, std::size_t count = Width,
                        bool enabled = true) const {
        if (count > Width)
            throw std::invalid_argument("Too many trajectory lanes");
        Samples out;
        out.count = count;
        if (!count)
            return out;
        std::array<double, Width> t{};
        for (std::size_t i = 0; i < count; ++i) {
            const auto *tr = trajectories[i];
            if (!tr || !std::isfinite(tr->T) || tr->T < 0 || !std::isfinite(times[i]) ||
                times[i] < -1e-10 || times[i] > tr->T + 1e-10)
                throw std::out_of_range("Invalid trajectory sample");
            t[i] = std::clamp(times[i], 0.0, tr->T);
        }
        std::array<double, 2 * Order * Width> coefficients{};
        for (int d = 0; d <= Order; ++d)
            for (int a = 0; a < Dim; ++a) {
                for (int k = d; k <= Trajectory::Degree; ++k)
                    for (std::size_t i = 0; i < count; ++i)
                        coefficients[(k - d) * count + i] =
                            trajectories[i]->coefficients(a, k) * (factorial(k) / factorial(k - d));
                polynomialBatch(coefficients.data(), Trajectory::Degree - d, t.data(), count,
                                out.values[d][a].data(), selectSIMDBackend(enabled));
            }
        return out;
    }
    std::array<State, Width> evaluate(const std::array<const Trajectory *, Width> &trajectories,
                                      const std::array<double, Width> &times) const {
        auto soa = evaluateSoA(trajectories, times);
        std::array<State, Width> out{};
        for (std::size_t i = 0; i < Width; ++i)
            out[i] = soa.state(i);
        return out;
    }
};
} // namespace rrtpff
