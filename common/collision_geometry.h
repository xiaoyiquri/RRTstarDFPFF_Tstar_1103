#pragma once
#include <array>

namespace rrtpff {
struct CollisionAABB {
    std::array<double, 3> lower{}, upper{};
};
} // namespace rrtpff
