#pragma once
#include "common/flat_trajectory.h"
#include "common/timer.h"
#include "common/types.h"
#include "dynobench/data_adapter.h"
#include "simd/physical_batch.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace rrtpff {
struct SampleGrid {
    std::size_t intervals{}, count{}, batches{};
    SampleGrid(double T, double dt, std::size_t maximum) {
        double n = std::ceil(T / dt);
        if (!std::isfinite(T) || T <= 0 || !std::isfinite(dt) || dt <= 0 || !std::isfinite(n) ||
            n < 1 || n + 1 > maximum)
            throw std::invalid_argument("Invalid or excessive discrete sample grid");
        intervals = static_cast<std::size_t>(n);
        count = intervals + 1;
        batches = (count + 7) / 8;
    }
    unsigned batch(std::size_t i, float times[8], std::size_t indices[8]) const {
        unsigned active = 0;
        for (std::size_t lane = 0; lane < 8; ++lane) {
            indices[lane] = i + lane * batches;
            if (indices[lane] < count) {
                active |= 1u << lane;
                times[lane] = float(double(indices[lane]) / intervals);
            } else
                times[lane] = 0;
        }
        return active;
    }
};
inline double chooseSmall(int n, int k) {
    double v = 1;
    for (int i = 1; i <= k; ++i)
        v *= double(n - k + i) / i;
    return v;
}
template <int Dim, int Order, class Metadata>
bool makeBernstein(const FlatTrajectory<Dim, Order, Metadata> &tr, BernsteinCurve &curve) {
    if (!std::isfinite(tr.T) || tr.T <= 0 || !tr.coefficients.allFinite())
        return false;
    curve = {};
    curve.dim = Dim;
    curve.order = Order;
    curve.derivative_order = 4;
    for (int d = 0; d <= curve.derivative_order; ++d)
        for (int k = 0; k < Dim; ++k) {
            const int degree = 2 * Order - 1 - d;
            if (degree < 0) continue; // Exact zero, including snap of a cubic.
            double power[8]{};
            for (int i = 0; i <= degree; ++i) {
                double factor = 1;
                for (int f = 1; f <= d; ++f)
                    factor *= i + f;
                power[i] = tr.coefficients(k, i + d) * factor * std::pow(tr.T, i);
            }
            for (int l = 0; l <= degree; ++l) {
                double b = 0;
                for (int i = 0; i <= l; ++i)
                    b += power[i] * chooseSmall(l, i) / chooseSmall(degree, i);
                curve.coefficients[d][k][l] = static_cast<float>(b);
                if (!std::isfinite(curve.coefficients[d][k][l]))
                    return false;
            }
        }
    return true;
}
template <int Dim, int Order, class Metadata>
bool validateDiscrete(const FlatTrajectory<Dim, Order, Metadata> &tr,
                      const PhysicalBatchConfig &config, int branch, const SphereGeometry &geometry,
                      double dt, std::size_t maximum, PlanResult &stats,
                      DynoSamples *samples = nullptr) {
    Timer timer;
    stats.validation_backend = "simd_discrete_float8";
    ++stats.simd_validation_queries;
    BernsteinCurve curve;
    if (!makeBernstein(tr, curve)) {
        ++stats.df_rejected;
        return false;
    }
    if (!std::isfinite(tr.T / dt) || std::ceil(tr.T / dt) + 1 > maximum) {
        ++stats.df_rejected;
        ++stats.df_failures[static_cast<std::size_t>(FeasibilityFailure::SampleLimit)];
        return false;
    }
    SampleGrid grid(tr.T, dt, maximum);
    if (samples) {
        samples->states.resize(grid.count);
        samples->controls.resize(grid.intervals);
        samples->times.resize(grid.count);
    }
    for (std::size_t b = 0; b < grid.batches; ++b) {
        float times[8];
        std::size_t indices[8];
        unsigned active = grid.batch(b, times, indices);
        PhysicalBatch out;
        curveBatch(curve, times, config, branch, active, geometry, out);
        ++stats.simd_validation_batches;
        for (int lane = 0; lane < 8; ++lane)
            if (active & (1u << lane)) {
                ++stats.simd_flat_samples;
                ++stats.simd_collision_samples;
            }
        stats.collision_check_count += out.sphere_tests;
        if (out.valid_mask != active) {
            ++stats.rejected_candidates;
            if (out.df_mask != active)
                ++stats.df_rejected;
            else if (out.state_mask != active)
                ++stats.state_bound_rejected;
            else if (out.control_mask != active)
                ++stats.control_bound_rejected;
            else
                ++stats.collision_rejected;
            stats.simd_time += timer.seconds();
            return false;
        }
        if (samples)
            for (int lane = 0; lane < 8; ++lane)
                if (active & (1u << lane)) {
                    const auto i = indices[lane];
                    samples->states[i].resize(config.state_dim);
                    for (int k = 0; k < config.state_dim; ++k)
                        samples->states[i](k) = out.states[k][lane];
                    samples->times[i] = tr.T * double(i) / grid.intervals;
                    if (i < grid.intervals) {
                        samples->controls[i].resize(config.control_dim);
                        for (int k = 0; k < config.control_dim; ++k)
                            samples->controls[i](k) = out.controls[k][lane];
                    }
                }
    }
    stats.simd_time += timer.seconds();
    return true;
}
} // namespace rrtpff
