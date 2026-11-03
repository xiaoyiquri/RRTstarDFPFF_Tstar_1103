/*
利用庞特里亚金极值原理（Pontryagin's Maximum Principle, PMP）
在微分平坦线性积分器链（Integrator Chain）上求解线性二次型（LQ）两点边值问题（TPBVP）的核心模板类
*/

#pragma once

#include "common/flat_trajectory.h"
#include "pff/pff_cost.h"
#include "pff/pff_integrator.h"

#include <Eigen/Cholesky>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace rrtpff {

struct PFFConfig {
    double min_time{.001}, max_time{5};
    bool free_terminal_time{true};
    double fixed_time{1};
    // Additional fixed-time optimal curves for physical feasibility search.
    // These sample times need not satisfy the unconstrained H(T)=0 condition.
    std::size_t additional_time_samples{0};
    // Rank exact fixed-time costs before allocating polynomial trajectories.
    bool defer_trajectories{false};
};

template <int Dim, int Order> struct TerminalConstraint {
    static constexpr int N = Dim * Order;
    using State = FlatState<Dim, Order>;

    State target{};
    std::array<bool, N> fixed{};

    static TerminalConstraint output(const typename State::Output &y) {
        TerminalConstraint c;
        c.target.setDerivative(0, y);
        for (int i = 0; i < Dim; ++i) {
            c.fixed[i] = true;
        }
        return c;
    }

    static TerminalConstraint full(const State &x) {
        TerminalConstraint c;
        c.target = x;
        c.fixed.fill(true);
        return c;
    }
};

template <int Dim, int Order> struct PFFTerminalSolution {
    bool success{false};
    typename FlatState<Dim, Order>::Vector eta = FlatState<Dim, Order>::Vector::Zero();
    double energy{0};
};

template <int Dim, int Order, typename Metadata = std::monostate> struct PFFResult {
    using State = FlatState<Dim, Order>;

    bool success{false};
    bool deferred{false};
    TerminalConstraint<Dim, Order> boundary{};
    // Cache the exact normalized-Gramian result, without allocating a polynomial.
    PFFTerminalSolution<Dim, Order> deferred_solution{};
    double T{0};
    double cost{std::numeric_limits<double>::infinity()};

    State terminal_state{};
    typename State::Vector terminal_costate = State::Vector::Zero();

    FlatTrajectory<Dim, Order, Metadata> trajectory{};

    std::vector<double> control_parameters;
    std::vector<double> polynomial_coefficients;
};

// Exact unconstrained LQ solution for arbitrary selected terminal coordinates
// of a homogeneous integrator chain. Physical constraints belong to the model.
// This is not a general affine-A matrix-exponential solver: LinearSystem is the
// representation, while this implementation deliberately exploits nilpotent A.
template <int Dim, int Order, typename Metadata = std::monostate> class PFFSolver {
  public:
    static constexpr int N = Dim * Order;

    using State = FlatState<Dim, Order>;
    using Output = typename State::Output;
    using Constraint = TerminalConstraint<Dim, Order>;
    using Trajectory = FlatTrajectory<Dim, Order, Metadata>;
    using Result = PFFResult<Dim, Order, Metadata>;
    using Chain = IntegratorChain<Dim, Order>;
    using Weight = typename Chain::Weight;
    using Energy = std::array<double, 2 * Order - 1>;

    PFFSolver(Weight R = Weight::Identity(), PFFConfig config = {})
        : R_(std::move(R)), config_(config) {
        if (!R_.allFinite() || !R_.isApprox(R_.transpose(), 1e-12)) {
            throw std::invalid_argument("PFF R must be finite and symmetric");
        }

        Eigen::LLT<Weight> factor(R_);
        if (factor.info() != Eigen::Success) {
            throw std::invalid_argument("PFF R must be positive definite");
        }

        inverseR_ = factor.solve(Weight::Identity());

        if (!std::isfinite(config.min_time) || !std::isfinite(config.max_time) ||
            !(config.min_time > 0) || config.max_time < config.min_time ||
            !std::isfinite(config.fixed_time) || config.fixed_time <= 0 ||
            config.additional_time_samples > 128) {
            throw std::invalid_argument("Invalid PFF time interval");
        }
    }

    const Weight &weight() const { return R_; }

    const PFFConfig &config() const { return config_; }

    struct PreparedConstraint {
        std::vector<int> rows;
        Eigen::MatrixXd gramian;
        Eigen::LDLT<Eigen::MatrixXd> factor;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inverse;
    };

    // Per-solver cache: one planner owns a solver. Parallel planners must own
    // independent solvers rather than share a mutable constraint cache.
    const PreparedConstraint &prepare(const Constraint &constraint) const {
        auto found = prepared_.find(constraint.fixed);
        if (found != prepared_.end())
            return *found->second;
        auto data = std::make_shared<PreparedConstraint>();
        data->rows = indices(constraint);
        if (data->rows.empty())
            throw std::invalid_argument("At least one terminal constraint required");
        data->gramian = projectedGramian(data->rows);
        data->factor.compute(data->gramian);
        if (data->factor.info() != Eigen::Success || !data->factor.isPositive())
            throw std::runtime_error("Singular projected Gramian");
        data->inverse =
            data->factor.solve(Eigen::MatrixXd::Identity(data->rows.size(), data->rows.size()));
        auto inserted = prepared_.emplace(constraint.fixed, std::move(data));
        return *inserted.first->second;
    }

    Eigen::MatrixXd energyBasis(const State &initial, const Constraint &constraint) const {
        const auto &rows = prepare(constraint).rows;
        Eigen::MatrixXd b = Eigen::MatrixXd::Zero(rows.size(), Order);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const int d = rows[row] / Dim, axis = rows[row] % Dim;
            b(row, Order - d - 1) += constraint.target.values(rows[row]);
            for (int j = d; j < Order; ++j)
                b(row, Order - j - 1) -= initial.values(j * Dim + axis) / factorial(j - d);
        }
        return b;
    }

    static std::vector<long double> timePolynomial(const Energy &energy) {
        std::vector<long double> polynomial(2 * Order + 1, 0);
        polynomial[2 * Order] = 1;
        for (int k = 1; k <= 2 * Order - 1; ++k)
            polynomial[2 * Order - k - 1] = -k * static_cast<long double>(energy[k - 1]);
        return polynomial;
    }

    using TerminalSolution = PFFTerminalSolution<Dim, Order>;

    TerminalSolution terminalSolution(const State &initial, const Constraint &constraint,
                                      double T) const {
        TerminalSolution result;

        if (!initial.finite() || !constraint.target.finite() || !std::isfinite(T) || T <= 0) {
            return result;
        }

        const auto rows = indices(constraint);
        if (rows.empty()) {
            return result;
        }

        const auto &prepared = prepare(constraint);
        const auto &G1 = prepared.gramian;
        const auto &factor = prepared.factor;

        const typename State::Vector bar = Chain::transition(T) * initial.values;

        Eigen::VectorXd h(rows.size());
        Eigen::VectorXd scales(rows.size());

        for (std::size_t i = 0; i < rows.size(); ++i) {
            scales(i) = std::pow(T, Order - .5 - rows[i] / Dim);
            h(i) = (constraint.target.values(rows[i]) - bar(rows[i])) / scales(i);
        }

        const Eigen::VectorXd zeta = factor.solve(h);

        if (!zeta.allFinite() || (G1 * zeta - h).norm() > 1e-8 * (1 + h.norm())) {
            return result;
        }

        for (std::size_t i = 0; i < rows.size(); ++i) {
            result.eta(rows[i]) = zeta(i) / scales(i);
        }
        result.energy = h.dot(zeta);
        result.success =
            result.eta.allFinite() && std::isfinite(result.energy) && result.energy >= -1e-10;
        return result;
    }

    Result fixedTime(const State &initial, const Constraint &constraint, double T) const {
        return buildTrajectory(initial, constraint, T, terminalSolution(initial, constraint, T));
    }

  private:
    Result buildTrajectory(const State &initial, const Constraint &constraint, double T,
                           const TerminalSolution &terminal) const {
        Result result;
        if (!terminal.success)
            return result;
        const auto &eta = terminal.eta;
        const auto &rows = prepare(constraint).rows;

        result.terminal_costate = -2 * eta;

        // free components have zero terminal costate
        result.control_parameters.reserve(Dim * Order);
        result.polynomial_coefficients.reserve(Dim * 2 * Order);
        auto &tr = result.trajectory;
        tr.T = T;
        tr.start = initial;

        for (int k = 0; k < Order; ++k) {
            tr.coefficients.col(k) = initial.derivative(k) / factorial(k);
        }

        for (int l = 0; l < Order; ++l) {
            Output u = Output::Zero();

            for (int k = l; k < Order; ++k) {
                u += eta.template segment<Dim>((Order - 1 - k) * Dim) *
                     (binomial(k, l) * std::pow(T, k - l) * (l % 2 ? -1.0 : 1.0) / factorial(k));
            }

            u = (inverseR_ * u).eval();
            tr.coefficients.col(Order + l) = u * (factorial(l) / factorial(Order + l));

            for (int a = 0; a < Dim; ++a) {
                result.control_parameters.push_back(u(a));
            }
        }

        if (!tr.coefficients.allFinite()) {
            return result;
        }

        tr.end = tr.evaluate(T);

        for (int row : rows) {
            const double target = constraint.target.values(row);
            if (std::abs(tr.end.values(row) - target) > 1e-7 * (1 + std::abs(target))) {
                return result;
            }
        }

        const double energy = terminal.energy;

        if (!std::isfinite(energy) || energy < -1e-10) {
            return result;
        }

        result.cost = tr.cost = PFFCost<Dim>::fromControlEnergy(T, std::max(0.0, energy));
        result.T = T;
        result.terminal_state = tr.end;

        for (int a = 0; a < Dim; ++a) {
            for (int k = 0; k < 2 * Order; ++k) {
                result.polynomial_coefficients.push_back(tr.coefficients(a, k));
            }
        }

        result.success = std::isfinite(result.cost) && tr.end.finite();
        return result;
    }

  public:
    Energy energyCoefficients(const State &initial, const Constraint &constraint) const {
        const auto b = energyBasis(initial, constraint);
        const Eigen::MatrixXd solved = prepare(constraint).factor.solve(b);

        Energy energy{};

        for (int k = 0; k < Order; ++k) {
            for (int l = 0; l < Order; ++l) {
                energy[k + l] += b.col(k).dot(solved.col(l));
            }
        }

        return energy;
    }

    std::vector<Result> candidates(const State &initial, const Constraint &constraint) const {
        std::vector<Result> results;

        if (!initial.finite() || !constraint.target.finite() || indices(constraint).empty()) {
            return results;
        }

        std::vector<double> times;

        if (config_.free_terminal_time) {
            const auto energy = energyCoefficients(initial, constraint);

            const auto polynomial = timePolynomial(energy);

            times = realRootsInInterval(polynomial, config_.min_time, config_.max_time);

            times.push_back(config_.min_time);
            times.push_back(config_.max_time);
        } else {
            times.push_back(config_.fixed_time);
        }

        return resultsAtTimes(initial, constraint, std::move(times));
    }

    std::vector<Result> resultsAtTimes(const State &initial, const Constraint &constraint,
                                       std::vector<double> times, bool include_extra = true) const {
        std::vector<Result> results;
        if (config_.free_terminal_time && include_extra) {
            for (std::size_t i = 1; i <= config_.additional_time_samples; ++i) {
                const double f = static_cast<double>(i) / (config_.additional_time_samples + 1);
                times.push_back(config_.min_time + f * (config_.max_time - config_.min_time));
            }
        }
        // NaNs do not define a strict ordering and must not enter std::sort.
        times.erase(std::remove_if(times.begin(), times.end(),
                                   [&](double t) {
                                       return !std::isfinite(t) || t < config_.min_time ||
                                              t > config_.max_time;
                                   }),
                    times.end());
        std::sort(times.begin(), times.end());
        times.erase(std::unique(times.begin(), times.end()), times.end());

        for (double T : times) {
            if (!std::isfinite(T) || T < config_.min_time || T > config_.max_time) {
                continue;
            }

            Result result;
            if (config_.defer_trajectories) {
                const auto terminal = terminalSolution(initial, constraint, T);
                if (terminal.success) {
                    result.cost =
                        PFFCost<Dim>::fromControlEnergy(T, std::max(0.0, terminal.energy));
                    result.T = T;
                    result.success = std::isfinite(result.cost);
                    result.deferred = true;
                    result.boundary = constraint;
                    result.deferred_solution = terminal;
                    result.trajectory.start = initial;
                }
            } else {
                result = fixedTime(initial, constraint, T);
            }

            if (result.success) {
                results.push_back(std::move(result));
            }
        }

        std::sort(results.begin(), results.end(),
                  [](const Result &a, const Result &b) { return a.cost < b.cost; });

        return results;
    }

    Result solve(const State &initial, const Constraint &constraint) const {
        auto results = candidates(initial, constraint);
        for (auto &result : results)
            if (materialize(initial, result))
                return std::move(result);
        return {};
    }

    bool materialize(const State &initial, Result &result) const {
        if (result.deferred) {
            // A deferred candidate is immutable until materialization. An explicit
            // different start requires a fresh solve, not a stale terminal cache.
            if ((initial.values.array() == result.trajectory.start.values.array()).all())
                result =
                    buildTrajectory(initial, result.boundary, result.T, result.deferred_solution);
            else
                result = fixedTime(initial, result.boundary, result.T);
        }
        return result.success;
    }

    double hamiltonian(const Result &result) const {
        const auto system = Chain::system();
        const auto &lambda = result.terminal_costate;

        const Output Btl = system.B.transpose() * lambda;

        return 1 + lambda.dot(system.A * result.terminal_state.values) -
               .25 * Btl.dot(inverseR_ * Btl);
    }

  private:
    static std::vector<int> indices(const Constraint &constraint) {
        std::vector<int> rows;

        for (int i = 0; i < N; ++i) {
            if (constraint.fixed[i]) {
                rows.push_back(i);
            }
        }

        return rows;
    }

    Eigen::MatrixXd projectedGramian(const std::vector<int> &rows) const {
        const auto G = Chain::gramian(1, inverseR_);
        Eigen::MatrixXd out(rows.size(), rows.size());

        for (std::size_t i = 0; i < rows.size(); ++i) {
            for (std::size_t j = 0; j < rows.size(); ++j) {
                out(i, j) = G(rows[i], rows[j]);
            }
        }

        return out;
    }

    Weight R_, inverseR_;
    PFFConfig config_;
    mutable std::map<std::array<bool, N>, std::shared_ptr<const PreparedConstraint>> prepared_;
};

} // namespace rrtpff
