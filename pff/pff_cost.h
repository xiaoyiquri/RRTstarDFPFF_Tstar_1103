/*

定义了 rrtpff::PFFCost 结构体，用于计算 PFF (Partial-Final-State-Free) 求解器中的代价函数
*/ 

#pragma once

#include <Eigen/Core>

#include <array>

namespace rrtpff {

template <int Dim>
struct PFFCost {
    using Weight = Eigen::Matrix<double, Dim, Dim>;

    Weight R = Weight::Identity();


	//cost 
    double runningCost(const Eigen::Matrix<double, Dim, 1>& u, double dt) const {
        return (1.0 + u.dot(R * u)) * dt;
    }

    double terminalCost() const {
        return 0;
    }

    static double fromControlEnergy(double T, double energy) {
        return T + energy;
    }

    template <std::size_t Size>
    static double totalCost(double T, const std::array<double, Size>& energy) {
        double J = T;
        double inv = 1 / T;
        double power = inv;

        for (double e : energy) {
            J += e * power;
            power *= inv;
        }

        return J;
    }
};

}  // namespace rrtpff
