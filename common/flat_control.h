#pragma once

#include "common/types.h"
#include <Eigen/Core>

namespace rrtpff {

template<int Dim, class Real = Scalar>
struct FlatControl {
    static_assert(Dim > 0, "Positive flat-control dimension required");

    using Vector = Eigen::Matrix<Real, Dim, 1>;

    Vector values = Vector::Zero();

    bool finite() const { return values.allFinite(); }

    Real& operator()(int index) { return values(index); }
    const Real& operator()(int index) const { return values(index); }

    operator const Vector&() const { return values; }
    operator Vector&() { return values; }
};

}  // namespace rrtpff
