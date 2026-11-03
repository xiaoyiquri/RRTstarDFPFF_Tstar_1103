#pragma once
#include "common/types.h"
#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

namespace rrtpff {
template<int Dim, int Order, class Real = Scalar>
struct FlatState {
    static_assert(Dim > 0 && Order > 0, "Positive integrator dimensions required");
    static constexpr int OutputDim = Dim;
    static constexpr int IntegratorOrder = Order;
    static constexpr int StateDim = Dim * Order;
    using Vector = Eigen::Matrix<Real, StateDim, 1>;
    using Output = Eigen::Matrix<Real, Dim, 1>;
    Vector values = Vector::Zero();
 // derivative-major: [y,dy,...]
    Output derivative(int d) const {
        if (d < 0 || d >= Order) {
            throw std::out_of_range("FlatState derivative order outside state");
        }
        return values.template segment<Dim>(d * Dim);
    }
    void setDerivative(int d, const Output& value) {
        if (d < 0 || d >= Order) {
            throw std::out_of_range("FlatState derivative order outside state");
        }
        values.template segment<Dim>(d * Dim) = value;
    }
    Output output() const { return derivative(0);
 }
    bool finite() const { return values.allFinite();
 }
};
}
