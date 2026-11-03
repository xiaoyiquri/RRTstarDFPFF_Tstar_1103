#pragma once
#include "simd/simd_flat_eval.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rrtpff {
struct CircleObstacle {
    double x{}, y{}, radius{};
};

// A rejection-only broad phase. Passing these samples never certifies a
// continuous trajectory. The caller must retain its exact/ground-truth check.
template <int Dim, int Order, typename Metadata = std::monostate, std::size_t Width = 8>
struct SIMDCollisionScreen {
    static_assert(Dim > 0 && Order >= 2 && Width > 0, "Screen requires position and velocity");
    using Trajectory = FlatTrajectory<Dim, Order, Metadata>;
    using Output = typename Trajectory::Output;
    struct Sphere {
        Output center = Output::Zero();
        double radius{0};
    };
    struct Limits {
        Output lower = Output::Constant(-std::numeric_limits<double>::infinity());
        Output upper = Output::Constant(std::numeric_limits<double>::infinity());
        double velocity = std::numeric_limits<double>::infinity();
        double control = std::numeric_limits<double>::infinity();
    };
    bool screen(const Trajectory &edge, const Limits &limits, const std::vector<Sphere> &obstacles,
                std::size_t samples = 33, bool interleaved = true, bool enabled = true,
                std::size_t *vector_samples = nullptr) const {
        if (samples < 2)
            throw std::invalid_argument("At least two collision samples required");
        if (!std::isfinite(edge.T) || edge.T <= 0 || !edge.coefficients.allFinite())
            return false;
        const auto backend = selectSIMDBackend(enabled);
        const std::size_t groups = (samples + Width - 1) / Width;
        for (std::size_t group = 0; group < groups; ++group) {
            std::array<const Trajectory *, Width> edges{};
            std::array<double, Width> times{};
            std::size_t count = 0;
            for (std::size_t lane = 0; lane < Width; ++lane) {
                const std::size_t index =
                    interleaved ? group + lane * groups : group * Width + lane;
                if (index >= samples)
                    continue;
                edges[count] = &edge;
                times[count++] = edge.T * (static_cast<double>(index) / (samples - 1));
            }
            auto flat = SIMDFlatEvaluator<Dim, Order, Metadata, Width>{}.evaluateSoA(
                edges, times, count, enabled);
            if (vector_samples && backend == SIMDBackend::AVX2)
                *vector_samples += (count / 4) * 4;
            // Only a confirmed scalar violation can reject a boundary lane.
            for (std::size_t i = 0; i < count; ++i) {
                bool suspect = false;
                for (int a = 0; a < Dim; ++a) {
                    const double x = flat.values[0][a][i];
                    suspect |= x < limits.lower(a) - 1e-8 || x > limits.upper(a) + 1e-8 ||
                               std::abs(flat.values[1][a][i]) > limits.velocity + 1e-8 ||
                               std::abs(flat.values[Order][a][i]) > limits.control + 1e-8;
                }
                if (suspect) {
                    const auto x = edge.evaluateState(times[i]);
                    const auto u = edge.evaluateControl(times[i]);
                    if ((x.output().array() < limits.lower.array() - 1e-8).any() ||
                        (x.output().array() > limits.upper.array() + 1e-8).any() ||
                        (x.derivative(1).array().abs() > limits.velocity + 1e-8).any() ||
                        (u.values.array().abs() > limits.control + 1e-8).any())
                        return false;
                }
            }
            std::array<double, Dim * Width> positions{};
            for (int a = 0; a < Dim; ++a)
                for (std::size_t i = 0; i < count; ++i)
                    positions[a * count + i] = flat.values[0][a][i];
            for (const auto &obstacle : obstacles) {
                std::array<unsigned char, Width> hit{};
                sphereBatch(positions.data(), obstacle.center.data(), Dim, obstacle.radius, count,
                            hit.data(), backend);
                for (std::size_t i = 0; i < count; ++i)
                    if (hit[i] &&
                        (edge.evaluateState(times[i]).output() - obstacle.center).squaredNorm() <=
                            obstacle.radius * obstacle.radius)
                        return false;
            }
        }
        return true;
    }
    std::array<bool, Width> collisionFree(const std::array<const Trajectory *, Width> &edges,
                                          const CircleObstacle &obstacle) const {
        static_assert(Dim == 2, "Circle screening requires two spatial coordinates");
        Sphere sphere;
        sphere.center << obstacle.x, obstacle.y;
        sphere.radius = obstacle.radius;
        std::array<bool, Width> result{};
        for (std::size_t i = 0; i < Width; ++i)
            result[i] = edges[i] && screen(*edges[i], Limits{}, std::vector<Sphere>{sphere});
        return result;
    }
};
} // namespace rrtpff
