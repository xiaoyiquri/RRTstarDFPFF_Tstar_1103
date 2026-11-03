#include "core/rrtstar.h"
#include "simd/discrete_validator.h"
#include "simd/simd_backend.h"
#include "systems/batch_config.h"
#include "systems/linear_model.h"
#include "tests/test_support.h"
#include <memory>
#include <random>
using namespace rrtpff;
template <class DF> DynoProblemData bounds() {
    DynoProblemData d;
    constexpr int nx = DF::PhysicalState::RowsAtCompileTime,
                  nu = DF::PhysicalControl::RowsAtCompileTime;
    d.lower_state = Eigen::VectorXd::Constant(nx, -100);
    d.upper_state = -d.lower_state;
    d.lower_control = Eigen::VectorXd::Constant(nu, -100);
    d.upper_control = -d.lower_control;
    return d;
}
template <class DF>
void parity(typename DF::Config c, const typename DF::FlatTrajectory::MetadataType &metadata,
            int branch) {
    auto data = bounds<DF>();
    auto config = makeBatchConfig<DF>(c, data);
    SphereGeometry geometry;
    geometry.dimension = config.position_dim;
    geometry.robot_radius = .1f;
    geometry.build();
    std::mt19937 gen(77);
    std::uniform_real_distribution<double> random(-1, 1);
    constexpr int Dim = DF::FlatState::OutputDim, Order = DF::FlatState::IntegratorOrder;
    for (int batch = 0; batch < 125; ++batch) {
        JetBatch jet;
        typename DF::PhysicalState x[8];
        typename DF::PhysicalControl u[8];
        bool valid[8];
        for (int lane = 0; lane < 8; ++lane) {
            typename DF::FlatState f;
            typename DF::FlatControl control;
            for (int d = 0; d <= Order; ++d)
                for (int k = 0; k < Dim; ++k) {
                    float value = float(.3 * random(gen));
                    if constexpr (std::is_same_v<DF, UnicycleDF>)
                        if (d == 1)
                            value += k == 0 ? .5f : 0;
                    if constexpr (std::is_same_v<DF, Quad3DDF>)
                        if (d == 0 && k == 3)
                            value = float(3.14159 * random(gen));
                    jet.values[d][k][lane] = value;
                    if (d < Order)
                        f.values(d * Dim + k) = value;
                    else
                        control.values(k) = value;
                }
            valid[lane] = DF::sampleFeasible(f, control, metadata, x[lane], u[lane], c);
        }
        PhysicalBatch out;
        physicalBatch(jet, config, branch, 255, geometry, out);
        for (int lane = 0; lane < 8; ++lane) {
            require(bool(out.valid_mask & (1u << lane)) == valid[lane],
                    "SIMD/scalar feasibility parity away from boundaries");
            if (!valid[lane])
                continue;
            for (int k = 0; k < config.state_dim; ++k) {
                if constexpr (std::is_same_v<DF, Quad3DDF>)
                    if (k >= 3 && k <= 6)
                        continue;
                double delta = out.states[k][lane] - x[lane](k);
                if constexpr (!std::is_same_v<DF, Quad3DDF>)
                    if (k == 2)
                        delta = std::remainder(delta, 2 * std::acos(-1.));
                require(std::abs(delta) < 3e-5, "SIMD/scalar state discrepancy");
            }
            if constexpr (std::is_same_v<DF, Quad3DDF>) {
                Eigen::Vector4d q;
                for (int k = 0; k < 4; ++k)
                    q(k) = out.states[k + 3][lane];
                require(std::min((q - x[lane].template segment<4>(3)).norm(),
                                 (q + x[lane].template segment<4>(3)).norm()) < 3e-5,
                        "quaternion rotation/sign parity");
            }
            for (int k = 0; k < config.control_dim; ++k)
                close(out.controls[k][lane], u[lane](k), 3e-5, "SIMD/scalar controls");
        }
    }
}
struct RejectFirst : LinearModel {
    using LinearModel::LinearModel;
    std::shared_ptr<int> attempts = std::make_shared<int>(0);
    bool trajectoryValid(const FlatTrajectory &, PlanResult &) const { return ++*attempts >= 20; }
};
template <class DF> void curveParity(typename DF::Config c) {
    constexpr int Dim = DF::FlatState::OutputDim, Order = DF::FlatState::IntegratorOrder;
    using Solver = PFFSolver<Dim, Order>;
    typename Solver::State a, b;
    a.values.setZero();
    b.values.setZero();
    b.values(0) = 1;
    b.values(1) = .5;
    auto config = makeBatchConfig<DF>(c, bounds<DF>());
    SphereGeometry geometry;
    geometry.robot_radius = .1f;
    geometry.dimension = config.position_dim;
    geometry.build();
    for (double T : {.25, 1., 5., 30.}) {
        PFFConfig pc;
        pc.min_time = .1;
        pc.max_time = 40;
        pc.free_terminal_time = false;
        pc.fixed_time = T;
        Solver solver(Solver::Weight::Identity(), pc);
        auto solved = solver.solve(a, Solver::Constraint::full(b));
        require(solved.success, "fixed full-boundary curve");
        BernsteinCurve curve;
        require(makeBernstein(solved.trajectory, curve), "full polynomial Bernstein conversion");
        for (int batch = 0; batch < 9; ++batch) {
            float times[8];
            for (int lane = 0; lane < 8; ++lane)
                times[lane] = float(batch * 8 + lane) / 71;
            PhysicalBatch out;
            curveBatch(curve, times, config, 1, 255, geometry, out);
            for (int lane = 0; lane < 8; ++lane) {
                const double t = T * times[lane];
                auto flat = solved.trajectory.evaluateState(t);
                auto control = solved.trajectory.evaluateControl(t);
                typename DF::PhysicalState x;
                typename DF::PhysicalControl u;
                bool state_ok = DF::flatToPhysicalState(flat, control, {}, x, c);
                bool control_ok = DF::flatToPhysicalControl(flat, control, {}, u, c);
                if (!state_ok || !control_ok)
                    continue;
                for (int k = 0; k < config.state_dim; ++k) {
                    if constexpr (std::is_same_v<DF, Quad3DDF>)
                        if (k >= 3 && k <= 6)
                            continue;
                    double diff = out.states[k][lane] - x(k);
                    if constexpr (std::is_same_v<DF, Quad2DDF>)
                        if (k == 2)
                            diff = std::remainder(diff, 2 * std::acos(-1.));
                    require(std::abs(diff) <= 3e-4 * (1 + std::abs(x(k))),
                            "full SIMD curve state agrees with double reference");
                }
                for (int k = 0; k < config.control_dim; ++k)
                    require(std::abs(out.controls[k][lane] - u(k)) <= 3e-4 * (1 + std::abs(u(k))),
                            "full SIMD curve controls agree with double reference");
            }
        }
    }
}
int main() {
    return runTest([] {
        require(simdRuntimeAVX2(), "AVX2 required: no scalar fallback in this test");
        UnicycleDF::Config uc;
        uc.allow_reverse = true;
        uc.velocity_lower = -2;
        uc.velocity_upper = 2;
        uc.max_velocity = 2;
        uc.max_omega = 10;
        uc.omega_lower = -10;
        uc.omega_upper = 10;
        UnicycleTrajectoryMetadata forward, reverse;
        require(UnicycleDF::makeMetadata(.5, forward, uc), "forward metadata");
        require(UnicycleDF::makeMetadata(-.5, reverse, uc), "reverse metadata");
        parity<UnicycleDF>(uc, forward, 1);
        parity<UnicycleDF>(uc, reverse, -1);
        Quad2DDF::Config qc;
        qc.max_acceleration = 100;
        qc.max_angular_acceleration = 100;
        parity<Quad2DDF>(qc, {}, 1);
        Quad3DDF::Config q3;
        q3.mass = 1;
        q3.inertia << .1, .1, .2;
        q3.max_motor = 1.5;
        q3.max_acceleration = 100;
        q3.max_angular_acceleration = 100;
        auto map = std::make_shared<WrenchControlMap>();
        map->inverse.diagonal() << 1 / 9.81, .05, .05, .05;
        map->lower << 0, -.1, -.1, -.1;
        map->upper << 1.5, .1, .1, .1;
        q3.control_map = map;
        parity<Quad3DDF>(q3, {}, 1);
        curveParity<Quad2DDF>(qc);
        curveParity<Quad3DDF>(q3);
        // Every index appears once, including both endpoints and partial batches.
        for (int n = 1; n < 100; ++n) {
            SampleGrid g(n, .999, 200);
            std::vector<int> seen(g.count);
            for (std::size_t b = 0; b < g.batches; ++b) {
                float t[8];
                std::size_t indices[8];
                auto mask = g.batch(b, t, indices);
                for (int k = 0; k < 8; ++k)
                    if (mask & (1u << k)) {
                        ++seen[indices[k]];
                        require(t[k] >= 0 && t[k] <= 1, "normalized time range");
                    }
            }
            for (int s : seen)
                require(s == 1, "interleaved exact sample coverage");
        }
        auto config = makeBatchConfig<UnicycleDF>(uc, bounds<UnicycleDF>());
        SphereGeometry geometry;
        geometry.robot_radius = .1f;
        geometry.spheres.push_back({{.5f, 0, 0}, .1f});
        geometry.build();
        JetBatch j;
        for (int k = 0; k < 8; ++k) {
            j.values[0][0][k] = float(k);
            j.values[1][0][k] = .5f;
        }
        j.values[0][0][3] = .5f;
        j.values[1][0][4] = 0;
        j.values[0][0][5] = std::numeric_limits<float>::quiet_NaN();
        PhysicalBatch result;
        physicalBatch(j, config, 1, 255, geometry, result);
        require(result.collision_mask == (1u << 3), "physical sphere collision lane");
        require(!(result.valid_mask & (1u << 4)), "zero speed singularity");
        require(!(result.valid_mask & (1u << 5)), "NaN fails closed");
        physicalBatch(j, config, 1, 7, geometry, result);
        require(result.valid_mask == 7, "inactive bad lanes masked");
        auto limited = config;
        limited.state_upper[0] = .5f;
        physicalBatch(j, limited, 1, 7, geometry, result);
        require(result.state_mask == 1, "SIMD physical position bounds");
        limited = config;
        limited.control_upper[0] = .4f;
        physicalBatch(j, limited, 1, 7, geometry, result);
        require(result.control_mask == 0, "SIMD actuator bounds");
        auto qconfig = makeBatchConfig<Quad3DDF>(q3, bounds<Quad3DDF>());
        SphereGeometry empty3;
        empty3.dimension = 3;
        empty3.robot_radius = .1f;
        empty3.build();
        JetBatch singular;
        singular.values[2][2][0] = -qconfig.gravity;
        physicalBatch(singular, qconfig, 1, 1, empty3, result);
        require(result.valid_mask == 0, "Quad3D zero thrust singularity");
        singular.values[2][0][0] = 1;
        physicalBatch(singular, qconfig, 1, 1, empty3, result);
        require(result.valid_mask == 0, "Quad3D heading/thrust parallel singularity");
        // Full trajectory evaluation + early rejection at a sampled interior point.
        UnicycleDF::FlatTrajectory edge;
        edge.T = 1;
        edge.coefficients.setZero();
        edge.coefficients(0, 1) = 1;
        edge.metadata = forward;
        edge.start = edge.evaluateState(0);
        edge.end = edge.evaluateState(1);
        BernsteinCurve curve;
        require(makeBernstein(edge, curve), "Bernstein conversion");
        PlanResult stats;
        require(!validateDiscrete(edge, config, 1, geometry, .01, 200, stats),
                "interior collision");
        geometry.spheres.clear();
        geometry.build();
        DynoSamples samples;
        require(validateDiscrete(edge, config, 1, geometry, .01, 200, stats, &samples),
                "valid curve");
        close(samples.states.front()(0), 0, 1e-6, "first endpoint");
        close(samples.states.back()(0), 1, 1e-6, "last endpoint");
        require(samples.states.size() == 101 && samples.controls.size() == 100,
                "export dimensions");
        // A candidate after the old K=16 budget must still be examined.
        LinearModel::Config lc;
        lc.start.values << 0, 0, 0, 0;
        lc.goal << 1, 0;
        lc.lower << -2, -2;
        lc.upper << 2, 2;
        RejectFirst model(lc);
        PlannerConfig pc;
        pc.timeout = 10;
        pc.max_iterations = 1;
        pc.goal_bias = 1;
        pc.range = 2;
        pc.simd_enabled = true;
        pc.simd_screen = false;
        pc.rewire_before_first_solution = false;
        PFFConfig pf;
        pf.min_time = .1;
        pf.max_time = 10;
        pf.additional_time_samples = 32;
        RRTstar<RejectFirst> planner(model, pc, PFFSolver<2, 2>(Eigen::Matrix2d::Identity(), pf));
        require(planner.solve().success && *model.attempts == 20,
                "all sorted candidates eligible without Top-K");
    });
}
