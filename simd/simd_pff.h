#pragma once
#include "pff/pff.h"
#include "simd/simd_roots.h"
#include <array>
#include <stdexcept>
#include <utility>
#include <variant>

namespace rrtpff {
template <int Dim, int Order, typename Metadata = std::monostate,
          std::size_t Width = SIMDBatchWidth>
struct BatchPFFResult {
    std::array<PFFResult<Dim, Order, Metadata>, Width> lane{};
    std::array<bool, Width> valid{};
};

template <int Dim, int Order, typename Metadata = std::monostate,
          std::size_t Width = SIMDBatchWidth>
class SIMD_PFF {
  public:
    using Solver = PFFSolver<Dim, Order, Metadata>;
    using State = typename Solver::State;
    using Constraint = typename Solver::Constraint;
    using Result = typename Solver::Result;
    explicit SIMD_PFF(Solver solver = Solver{}) : solver_(std::move(solver)) {}
    bool deferred() const { return solver_.config().defer_trajectories; }
    bool materialize(const State &initial, Result &result) const {
        return solver_.materialize(initial, result);
    }

    // Retain every candidate duration, including interval boundaries. Returning
    // only the cheapest unconstrained root can discard the feasible solution.
    std::array<std::vector<Result>, Width>
    candidatesBatch(const std::array<State, Width> &parents,
                    const std::array<Constraint, Width> &constraints, std::size_t count = Width,
                    bool enabled = true, std::size_t *vector_lanes = nullptr) const {
        if (count > Width)
            throw std::invalid_argument("Too many PFF lanes");
        std::array<std::vector<Result>, Width> out{};
        const auto backend = selectSIMDBackend(enabled);
        if (!count)
            return out;
        bool compatible = backend == SIMDBackend::AVX2 && count >= 4;
        for (std::size_t i = 0; i < count; ++i)
            compatible &= parents[i].finite() && constraints[i].target.finite() &&
                          constraints[i].fixed == constraints[0].fixed;
        bool has_fixed = false;
        for (bool fixed : constraints[0].fixed)
            has_fixed |= fixed;
        compatible &= has_fixed;
        if (!compatible || !solver_.config().free_terminal_time) {
            for (std::size_t i = 0; i < count; ++i)
                out[i] = solver_.candidates(parents[i], constraints[i]);
            return out;
        }
        const auto &prepared = solver_.prepare(constraints[0]);
        const int rows = static_cast<int>(prepared.rows.size());
        std::vector<double> basis(rows * Order * count), energy((2 * Order - 1) * count);
        for (std::size_t i = 0; i < count; ++i) {
            auto b = solver_.energyBasis(parents[i], constraints[i]);
            for (int r = 0; r < rows; ++r)
                for (int k = 0; k < Order; ++k)
                    basis[(r * Order + k) * count + i] = b(r, k);
        }
        energyBatch(basis.data(), prepared.inverse.data(), rows, Order, count, energy.data(),
                    backend);
        std::array<std::vector<long double>, Width> polynomials{};
        for (std::size_t i = 0; i < count; ++i) {
            typename Solver::Energy e{};
            for (int k = 0; k < 2 * Order - 1; ++k)
                e[k] = energy[k * count + i];
            polynomials[i] = Solver::timePolynomial(e);
        }
        auto times = batchRoots<Width>(polynomials, solver_.config().min_time,
                                       solver_.config().max_time, count, backend);
        if (vector_lanes)
            *vector_lanes += (count / 4) * 4;
        for (std::size_t i = 0; i < count; ++i) {
            times[i].push_back(solver_.config().min_time);
            times[i].push_back(solver_.config().max_time);
            // The same reference computes exact costs. In deferred mode full
            // polynomial construction occurs only when the planner needs it.
            out[i] = solver_.resultsAtTimes(parents[i], constraints[i], std::move(times[i]));
        }
        return out;
    }
    BatchPFFResult<Dim, Order, Metadata, Width>
    evaluateBatch(const std::array<State, Width> &parents,
                  const std::array<Constraint, Width> &constraints, std::size_t count = Width,
                  bool enabled = true) const {
        auto candidates = candidatesBatch(parents, constraints, count, enabled);
        BatchPFFResult<Dim, Order, Metadata, Width> out;
        for (std::size_t i = 0; i < count; ++i)
            for (auto &candidate : candidates[i])
                if (materialize(parents[i], candidate)) {
                    out.lane[i] = std::move(candidate);
                    out.valid[i] = true;
                    break;
                }
        return out;
    }

  private:
    Solver solver_;
};
} // namespace rrtpff
