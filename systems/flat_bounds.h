#pragma once
#include "common/flat_trajectory.h"
#include "simd/simd_flat_eval.h"
#include <array>

namespace rrtpff {
// Rejection-only broad phase. It never certifies physical feasibility or
// collision freedom; dense DF/DynoBench checking still follows on acceptance.
template <int Dim, int Order, class Metadata>
bool flatBoundsScreen(const FlatTrajectory<Dim, Order, Metadata> &trajectory,
                      const typename FlatState<Dim, Order>::Output &lower,
                      const typename FlatState<Dim, Order>::Output &upper, int position_dim,
                      double max_speed, double max_acceleration, std::size_t intervals, bool simd,
                      PlanResult &stats) {
    if (intervals < 1 || intervals > 1000000 || position_dim < 1 || position_dim > Dim ||
        !std::isfinite(trajectory.T) || trajectory.T <= 0 || !trajectory.coefficients.allFinite())
        return false;
    constexpr std::size_t Width = 8;
    using Trajectory = FlatTrajectory<Dim, Order, Metadata>;
    for (std::size_t offset = 0; offset <= intervals; offset += Width) {
        const auto count = std::min(Width, intervals + 1 - offset);
        std::array<const Trajectory *, Width> curves{};
        std::array<double, Width> times{};
        for (std::size_t i = 0; i < count; ++i) {
            curves[i] = &trajectory;
            times[i] = trajectory.T * static_cast<double>(offset + i) / intervals;
        }
        auto samples = SIMDFlatEvaluator<Dim, Order, Metadata, Width>{}.evaluateSoA(curves, times,
                                                                                    count, simd);
        if (simd && simdRuntimeAVX2())
            stats.simd_flat_samples += (count / 4) * 4;
        for (std::size_t i = 0; i < count; ++i) {
            const auto flat = samples.state(i);
            const auto position = flat.output();
            if (!flat.finite())
                return false;
            // Yaw is an unwrapped flat coordinate, not a physical position
            // bound. The yaw sampling interval is not an actuator constraint.
            for (int d = 0; d < position_dim; ++d)
                if (position(d) < lower(d) - 1e-9 || position(d) > upper(d) + 1e-9)
                    return false;
            if (flat.derivative(1).head(position_dim).norm() > max_speed + 1e-9)
                return false;
            if constexpr (Order > 2)
                if (flat.derivative(2).head(position_dim).norm() > max_acceleration + 1e-9)
                    return false;
        }
    }
    return true;
}
} // namespace rrtpff
