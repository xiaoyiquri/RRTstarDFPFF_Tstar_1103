#pragma once
#include "common/collision_geometry.h"
#include "common/flat_trajectory.h"
#include "common/types.h"
#include "simd/simd_backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace rrtpff {
// Only rejects collisions proved by an inscribed robot sphere. Passing this
// screen NEVER bypasses physical reconstruction or actual DynoBench/FCL checks.
template <int Dim, int Order, class Metadata>
bool robotCollisionScreen(const FlatTrajectory<Dim, Order, Metadata> &edge, int position_dim,
                          double inner_radius, const std::vector<CollisionAABB> &obstacles,
                          std::size_t intervals, bool simd, PlanResult &stats) {
    if (!(inner_radius > 1e-9) || obstacles.empty() || intervals == 0)
        return true; // Unsupported geometry is left to the authoritative backend.
    if (!std::isfinite(edge.T) || edge.T <= 0 || !edge.coefficients.allFinite())
        return false;
    if (position_dim < 1 || position_dim > Dim || position_dim > 3 || intervals > 1000000)
        return false;
    constexpr std::size_t Width = SIMDBatchWidth;
    const auto backend = selectSIMDBackend(simd);
    const double radius = inner_radius - 1e-9;
    for (std::size_t offset = 0; offset <= intervals; offset += Width) {
        const auto count = std::min(Width, intervals + 1 - offset);
        // The paper's collision path uses eight float lanes. Keep the scalar
        // double path as the reference fallback when AVX2 is unavailable.
        if (backend == SIMDBackend::AVX2) {
            std::array<float, Width> times{};
            std::array<float, 2 * Order * Width> coefficients{};
            std::array<float, 3 * Width> position{};
            for (std::size_t i = 0; i < count; ++i)
                times[i] = static_cast<float>(edge.T * static_cast<double>(offset + i) / intervals);
            for (int axis = 0; axis < position_dim; ++axis) {
                for (int k = 0; k < 2 * Order; ++k)
                    std::fill_n(coefficients.data() + k * count, count,
                                static_cast<float>(edge.coefficients(axis, k)));
                polynomialBatchFloat(coefficients.data(), 2 * Order - 1, times.data(), count,
                                     position.data() + axis * count, backend);
            }
            stats.simd_collision_samples += (count / Width) * Width;
            for (const auto &obstacle : obstacles) {
                std::array<float, 3> lower{}, upper{};
                for (int axis = 0; axis < position_dim; ++axis) {
                    lower[axis] = static_cast<float>(obstacle.lower[axis]);
                    upper[axis] = static_cast<float>(obstacle.upper[axis]);
                }
                std::array<unsigned char, Width> hit{};
                sphereAABBBatchFloat(position.data(), lower.data(), upper.data(), position_dim,
                                     static_cast<float>(radius), count, hit.data(), backend);
                if (std::any_of(hit.begin(), hit.begin() + count,
                                [](auto h) { return h != 0; }))
                    return false;
            }
        } else {
            std::array<double, Width> times{};
            std::array<double, 2 * Order * Width> coefficients{};
            std::array<double, 3 * Width> position{};
            for (std::size_t i = 0; i < count; ++i)
                times[i] = edge.T * static_cast<double>(offset + i) / intervals;
            for (int axis = 0; axis < position_dim; ++axis) {
                for (int k = 0; k < 2 * Order; ++k)
                    std::fill_n(coefficients.data() + k * count, count, edge.coefficients(axis, k));
                polynomialBatch(coefficients.data(), 2 * Order - 1, times.data(), count,
                                position.data() + axis * count, backend);
            }
            for (const auto &obstacle : obstacles) {
                std::array<unsigned char, Width> hit{};
                sphereAABBBatch(position.data(), obstacle.lower.data(), obstacle.upper.data(),
                                position_dim, radius, count, hit.data(), backend);
                if (std::any_of(hit.begin(), hit.begin() + count,
                                [](auto h) { return h != 0; }))
                    return false;
            }
        }
    }
    return true;
}
} // namespace rrtpff
