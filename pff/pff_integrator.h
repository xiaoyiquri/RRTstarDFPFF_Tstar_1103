/*
定义了线性系统结构体 LinearSystem 以及积分器链（Integrator Chain）工具类 IntegratorChain，
它为 PFF 求解器（PFFSolver）提供了底层的状态转移矩阵（State Transition Matrix）
与可控性格拉姆矩阵（Controllability Gramian）的解析计算
*/

#pragma once

#include "common/flat_state.h"
#include "common/math_utils.h"

namespace rrtpff {

template <int StateDim, int ControlDim> struct LinearSystem {
    Eigen::Matrix<double, StateDim, StateDim> A = Eigen::Matrix<double, StateDim, StateDim>::Zero();
    Eigen::Matrix<double, StateDim, ControlDim> B =
        Eigen::Matrix<double, StateDim, ControlDim>::Zero();
    Eigen::Matrix<double, StateDim, 1> c = Eigen::Matrix<double, StateDim, 1>::Zero();
};

template <int Dim, int Order> class IntegratorChain {
  public:
    static constexpr int N = Dim * Order;

    using Matrix = Eigen::Matrix<double, N, N>;
    using Weight = Eigen::Matrix<double, Dim, Dim>;

    static LinearSystem<N, Dim> system() {
        LinearSystem<N, Dim> out;

        for (int k = 0; k < Order - 1; ++k) {
            out.A.template block<Dim, Dim>(k * Dim, (k + 1) * Dim).setIdentity();
        }

        out.B.template bottomRows<Dim>().setIdentity();
        return out;
    }

    static Matrix transition(double T) {
        Matrix phi = Matrix::Zero();

        for (int i = 0; i < Order; ++i) {
            for (int j = i; j < Order; ++j) {
                phi.template block<Dim, Dim>(i * Dim, j * Dim) =
                    Weight::Identity() * (std::pow(T, j - i) / factorial(j - i));
            }
        }

        return phi;
    }

    static Matrix gramian(double T, const Weight &inverseR) {
        Matrix G;

        for (int i = 0; i < Order; ++i) {
            for (int j = 0; j < Order; ++j) {
                int n = 2 * Order - 1 - i - j;

                G.template block<Dim, Dim>(i * Dim, j * Dim) =
                    inverseR *
                    (std::pow(T, n) / (factorial(Order - 1 - i) * factorial(Order - 1 - j) * n));
            }
        }

        return G;
    }
};

} // namespace rrtpff
