#pragma once
#include <Eigen/Core>
namespace rrtpff {
template <int Dim> struct SamplingRegion {
    Eigen::Matrix<double, Dim, 1> lower, upper;
};
} // namespace rrtpff
