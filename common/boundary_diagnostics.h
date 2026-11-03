#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rrtpff {
struct PhysicalBoundaryError {
    double position{}, velocity{}, attitude{}, omega{};
    double maximum() const { return std::max({position, velocity, attitude, omega}); }
};
// Attitude errors are radians and quaternion-sign invariant. Other errors are
// Euclidean norms in their own physical units, reported in separate columns.
inline PhysicalBoundaryError physicalBoundaryError(const Eigen::VectorXd &a,
                                                   const Eigen::VectorXd &b) {
    if (a.size() != b.size() || !a.allFinite() || !b.allFinite())
        throw std::invalid_argument("Invalid physical boundary diagnostic inputs");
    PhysicalBoundaryError e;
    const double pi = std::acos(-1.);
    if (a.size() == 13) {
        e.position = (a.head<3>() - b.head<3>()).norm();
        e.velocity = (a.segment<3>(7) - b.segment<3>(7)).norm();
        Eigen::Quaterniond qa(a(6), a(3), a(4), a(5)), qb(b(6), b(3), b(4), b(5));
        if (qa.norm() == 0 || qb.norm() == 0)
            throw std::invalid_argument("Zero quaternion at physical boundary");
        qa.normalize(); qb.normalize();
        const Eigen::Quaterniond delta = qa.conjugate() * qb;
        e.attitude = 2 * std::atan2(delta.vec().norm(), std::abs(delta.w()));
        e.omega = (a.tail<3>() - b.tail<3>()).norm();
    } else if (a.size() == 6) {
        e.position = (a.head<2>() - b.head<2>()).norm();
        e.velocity = (a.segment<2>(3) - b.segment<2>(3)).norm();
        e.attitude = std::abs(std::remainder(a(2) - b(2), 2 * pi));
        e.omega = std::abs(a(5) - b(5));
    } else if (a.size() == 3) {
        e.position = (a.head<2>() - b.head<2>()).norm();
        e.attitude = std::abs(std::remainder(a(2) - b(2), 2 * pi));
        // Unicycle1 speed and omega are controls, not physical state entries.
    } else throw std::invalid_argument("Unknown physical state diagnostic layout");
    return e;
}
struct BoundaryDiagnostic {
    // 0=start, 1=junction, 2=goal; edge_index is the incoming/right edge.
    int kind{};
    std::size_t edge_index{};
    double time{};
    double flat_position_error{}, flat_velocity_error{};
    PhysicalBoundaryError physical;
};
} // namespace rrtpff
