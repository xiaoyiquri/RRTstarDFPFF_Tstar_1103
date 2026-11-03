#include "common/timer.h"
#include "simd/simd_pff.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace rrtpff;
struct Measurement {
    double seconds = 0, checksum = 0;
    std::size_t queries = 0, candidates = 0, curves = 0;
};
template <int Dim, int Order> Measurement measure(bool deferred, bool all) {
    using S = PFFSolver<Dim, Order>;
    PFFConfig c;
    c.min_time = .3;
    c.max_time = 6;
    c.additional_time_samples = 12;
    c.defer_trajectories = deferred;
    SIMD_PFF<Dim, Order> solver(S(S::Weight::Identity(), c));
    std::array<typename S::State, 8> starts{};
    std::array<typename S::Constraint, 8> targets{};
    for (std::size_t i = 0; i < 8; ++i) {
        typename S::State end;
        for (int d = 0; d < Dim * Order; ++d) {
            starts[i].values(d) = .05 * std::sin(i + d + .1);
            end.values(d) = .1 * std::cos(i + d + .7);
        }
        targets[i] = S::Constraint::full(end);
    }
    Measurement out;
    Timer timer;
    for (int repeat = 0; repeat < 160; ++repeat) {
        // Vary the workload without changing its difficulty across the two modes.
        starts[0].values(0) += 1e-6;
        auto batch = solver.candidatesBatch(starts, targets, 8, true);
        out.queries += 8;
        for (std::size_t i = 0; i < 8; ++i) {
            out.candidates += batch[i].size();
            if (!deferred)
                out.curves += batch[i].size();
            const auto used = all ? batch[i].size() : std::min<std::size_t>(1, batch[i].size());
            for (std::size_t k = 0; k < used; ++k) {
                if (batch[i][k].deferred)
                    ++out.curves;
                if (!solver.materialize(starts[i], batch[i][k]))
                    throw std::runtime_error("materialization failed");
                out.checksum += batch[i][k].cost + batch[i][k].trajectory.coefficients(0, Order);
            }
        }
    }
    out.seconds = timer.seconds();
    return out;
}
template <int Dim, int Order> void report() {
    for (bool all : {false, true}) {
        std::array<Measurement, 3> eager{}, lazy{};
        for (int trial = 0; trial < 3; ++trial) {
            // Alternate execution order to reduce warm-cache/order bias.
            if (trial % 2) {
                lazy[trial] = measure<Dim, Order>(true, all);
                eager[trial] = measure<Dim, Order>(false, all);
            } else {
                eager[trial] = measure<Dim, Order>(false, all);
                lazy[trial] = measure<Dim, Order>(true, all);
            }
            if (std::abs(eager[trial].checksum - lazy[trial].checksum) >
                1e-9 * (1 + std::abs(eager[trial].checksum)))
                throw std::runtime_error("benchmark checksum mismatch");
        }
        auto byTime = [](const auto &a, const auto &b) { return a.seconds < b.seconds; };
        std::sort(eager.begin(), eager.end(), byTime);
        std::sort(lazy.begin(), lazy.end(), byTime);
        const auto &a = eager[1];
        const auto &b = lazy[1];
        std::cout << Dim << ',' << Order << ',' << (all ? "all_visited" : "first_visited") << ','
                  << a.queries << ',' << a.candidates << ',' << a.curves << ',' << b.curves << ','
                  << a.seconds << ',' << b.seconds << ',' << a.seconds / b.seconds << '\n';
    }
}
int main() {
    try {
        std::cout << std::setprecision(9);
        std::cerr << "PFF microbenchmark only; k=8; AVX2=" << simdRuntimeAVX2()
                  << "; median of 3. Not six-environment planning time.\n";
        std::cout << "dim,order,workload,queries,candidates,eager_curves,lazy_curves,eager_seconds,"
                     "lazy_seconds,speedup\n";
        report<2, 2>();
        report<2, 4>();
        report<4, 4>();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
