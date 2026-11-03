#pragma once
#include "common/flat_state.h"
#include "common/flat_control.h"
#include "common/math_utils.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <variant>

namespace rrtpff {
template<int Dim, int Order, typename Metadata = std::monostate, class Real = Scalar>
struct FlatTrajectory {
    using State=FlatState<Dim,Order,Real>;
    using Control=FlatControl<Dim,Real>;
    using Output=typename State::Output;
    using MetadataType=Metadata;
    static constexpr int Degree=2*Order-1;
    Real T{0}, cost{0};
    bool normalized{false};
    Eigen::Matrix<Real,Dim,2*Order> coefficients=Eigen::Matrix<Real,Dim,2*Order>::Zero();
    State start{}, end{};
    Metadata metadata{};
    Output derivative(Real t,int d) const {
        if(!std::isfinite(t)||t < -Real(1e-6)||t > T+Real(1e-6) || d<0)
            throw std::out_of_range("Trajectory evaluation outside its domain");
        if (d > Degree) return Output::Zero();
        t=std::clamp(t,Real(0),T);
        if (normalized) t /= T;
        Output result=Output::Zero();
        for(int k=Degree;k>=d;--k)
            {
            Real factor = Real(1);
            for (int j=0;j<d;++j) factor *= Real(k-j);
            result=result*t+coefficients.col(k)*factor;
        }
        if (normalized) for(int j=0;j<d;++j) result /= T;
        return result;
    }
    State evaluate(Real t) const {
        State result;
        for(int d=0;d<Order;++d) {
            result.setDerivative(d, derivative(t, d));
        }
        return result;
    }

    State evaluateState(Real t) const { return evaluate(t); }

    Control evaluateControl(Real t) const {
        Control result;
        result.values = derivative(t, Order);
        return result;
    }

    Output control(Real t) const {return derivative(t,Order);
}
    Output velocity(Real t) const {return derivative(t,1);
}
    Output acceleration(Real t) const {return derivative(t,2);
}
    Output jerk(Real t) const {return derivative(t,3);
}
    Output snap(Real t) const {return derivative(t,4);
}
    template<std::size_t Width>
    std::array<State,Width> evaluateBatch(const std::array<Real,Width>& times) const {
        std::array<State,Width> out{};
        for(std::size_t i=0;i<Width;++i) out[i]=evaluate(times[i]);
        return out;
 // scalar reference; SIMD is a separate later layer
    }
};
}
