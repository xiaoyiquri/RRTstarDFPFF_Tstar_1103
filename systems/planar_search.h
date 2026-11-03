#pragma once
#include "common/sampling_region.h"
#include <Eigen/Core>
#include <vector>

namespace rrtpff {
// Explicit extra boundary conditions, not missing LiftData. Ordinary PFF
// expansions still leave the unselected derivatives free.
struct Quad2DTerminalHint {
    SamplingRegion<2> region;
    Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
    double theta{0}, force_ratio{1};
};
struct PlanarSearchConfig {
    bool turn_limited_sampling{false}, rest_hint{false};
    double terminal_speed{0};
    double turn_safety_factor{0.7};
    std::vector<double> duration_factors;
    std::vector<Quad2DTerminalHint> tilt_hints;
};
} // namespace rrtpff
