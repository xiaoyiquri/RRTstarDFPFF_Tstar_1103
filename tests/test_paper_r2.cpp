#include "tests/test_support.h"
#include "systems/robot_factory.h"
#include "simd/discrete_validator.h"
#include <memory>
using namespace rrtpff;

struct FixtureAdapter {
    std::shared_ptr<DynoProblemData> data = std::make_shared<DynoProblemData>();
    SphereGeometry spheres;
    const DynoProblemData &problem() const { return *data; }
    const SphereGeometry &geometry() const { return spheres; }
};
FixtureAdapter fixture(int nx, int nu, int dim) {
    FixtureAdapter a;
    a.data->start = Eigen::VectorXd::Zero(nx);
    a.data->goal = a.data->start;
    a.data->lower_position = Eigen::VectorXd::Constant(dim, -100);
    a.data->upper_position = -a.data->lower_position;
    a.data->lower_state = Eigen::VectorXd::Constant(nx, -100);
    a.data->upper_state = -a.data->lower_state;
    a.data->lower_control = Eigen::VectorXd::Constant(nu, -100);
    a.data->upper_control = -a.data->lower_control;
    a.spheres.dimension = dim;
    a.spheres.robot_radius = .1f;
    a.spheres.build();
    return a;
}
template <class R2, class Full> void cubicParity(typename Full::Config c) {
    constexpr int Dim = R2::FlatState::OutputDim;
    typename R2::FlatTrajectory tr;
    tr.T = 2;
    tr.coefficients(0, 1) = .3;
    tr.coefficients(0, 2) = .02;
    tr.coefficients(0, 3) = .01; // Nonzero jerk must survive r=2 conversion.
    tr.coefficients(1, 0) = 1;
    tr.start = tr.evaluate(0);
    tr.end = tr.evaluate(tr.T);
    require(tr.snap(1).isZero(), "cubic snap is exactly zero");
    auto adapter = fixture(Full::PhysicalState::RowsAtCompileTime,
                           Full::PhysicalControl::RowsAtCompileTime, Dim == 4 ? 3 : 2);
    const auto config = makeBatchConfig<R2>(c, adapter.problem());
    BernsteinCurve curve;
    require(makeBernstein(tr, curve), "r2 curve conversion");
    close(curve.coefficients[3][0][0], .06, 1e-6, "jerk retained");
    float times[8]{0, .1f, .2f, .3f, .4f, .6f, .8f, 1};
    PhysicalBatch out;
    curveBatch(curve, times, config, 1, 255, adapter.geometry(), out);
    for (int lane = 0; lane < 8; ++lane) {
        const double t = times[lane] * tr.T;
        typename Full::FlatState full;
        for (int d = 0; d < 4; ++d) full.setDerivative(d, tr.derivative(t, d));
        typename Full::FlatControl snap;
        snap.values = tr.snap(t);
        typename Full::PhysicalState x;
        typename Full::PhysicalControl u;
        require(Full::sampleFeasible(full, snap, {}, x, u, c), "double reference feasible");
        require(out.valid_mask & (1u << lane), "SIMD r2 feasible");
        Eigen::VectorXd actual(x.size());
        for (int k = 0; k < x.size(); ++k) actual(k) = out.states[k][lane];
        const auto error = physicalBoundaryError(x, actual);
        require(error.maximum() < 1e-4, "r2/full-reference state parity");
        for (int k = 0; k < u.size(); ++k)
            close(out.controls[k][lane], u(k), 1e-4, "r2/reference control parity");
    }
}
int main() {
    return runTest([] {
        static_assert(Quad2DR2DF::FlatState::StateDim == 4);
        static_assert(Quad3DR2DF::FlatState::StateDim == 8);
        static_assert(Quad2DR2DF::FlatTrajectory::Degree == 3);
        static_assert(Quad3DR2DF::FlatTrajectory::Degree == 3);
        auto no_guidance = YAML::Load("profile: paper_fair\nsystem: quad2d\ndf: {mode: paper_r2}\nsampling: {guidance_probability: 0}\n");
        validatePaperFairConfig(no_guidance);
        for (const char *bad : {
            "profile: paper_fair\nsampling: {guidance_probability: 0.2}",
            "profile: paper_fair\nsampling: {guidance_probability: 0, regions: []}",
            "profile: paper_fair\nplanar_search: {tilt_hints: []}",
            "system: quad2d\ndf: {mode: paper_r2}\nlifting: {start: {jerk: [0,0]}}",
            "df: {mode: invalid}"}) {
            bool rejected = false;
            try { validatePaperFairConfig(YAML::Load(bad)); }
            catch (const std::exception &) { rejected = true; }
            require(rejected, "forbidden profile input rejected");
        }
        Quad2DDFConfig c2;
        c2.max_thrust = 100; c2.max_velocity = 20; c2.max_motor = 100;
        cubicParity<Quad2DR2DF, Quad2DDF>(c2);
        Quad3DDFConfig c3;
        c3.max_motor = 100;
        auto map = std::make_shared<WrenchControlMap>();
        map->lower.setConstant(-100); map->upper.setConstant(100);
        c3.control_map = map;
        cubicParity<Quad3DR2DF, Quad3DDF>(c3);

        // Yaw rate is calculated from Rdot=R*skew(omega), not set to body wz.
        Quad3DR2DF::PhysicalState tilted = Quad3DR2DF::PhysicalState::Zero();
        Eigen::Quaterniond q(Eigen::AngleAxisd(.4, Eigen::Vector3d::UnitX()));
        tilted.segment<4>(3) = q.coeffs(); tilted(12) = 1;
        Quad3DR2DF::FlatState lifted;
        require(Quad3DR2DF::stateToFlat(tilted, {}, lifted, c3), "r2 tilted lift");
        close(lifted.derivative(1)(3), std::cos(.4), 1e-12, "Euler yaw rate from body omega");

        // A': a genuine attitude/omega jump is reported, not rejected or hidden.
        auto a = fixture(6, 2, 2);
        RobotValidationConfig v;
        v.df_mode = "paper_r2"; v.max_step = .05; v.endpoint_tolerance = 1e-3;
        Quad2DR2DF::FlatTrajectory first, second;
        first.T = second.T = 1;
        first.coefficients(0, 2) = 1;
        second.coefficients(0, 0) = 1; second.coefficients(0, 1) = 2;
        second.coefficients(0, 2) = 1; second.coefficients(0, 3) = .1;
        first.start = first.evaluate(0); first.end = first.evaluate(1);
        second.start = second.evaluate(0); second.end = second.evaluate(1);
        // Change acceleration at the shared node while retaining y/dy.
        second.coefficients(0, 2) = .5;
        second.end = second.evaluate(1);
        a.data->goal(0) = 3.6; a.data->goal(3) = 3.3;
        DynoSystemModel<Quad2DR2DF, FixtureAdapter> reduced(a, {}, {}, c2, v);
        DynoSamples samples; PlanResult stats;
        require(reduced.validatePath({first, second}, samples, stats).success, "A' allows diagnosed attitude jump");
        require(stats.boundary_diagnostics.size() == 3, "both endpoints plus junction reported");
        require(stats.boundary_diagnostics[1].physical.attitude > .05, "junction attitude jump recorded");
        require(stats.boundary_diagnostics[1].physical.omega > .01, "junction omega jump recorded");
        require(!stats.physical_boundary_match && stats.reduced_order_result, "reduced result explicitly marked");
        second.start.values(0) += .1;
        require(!reduced.validatePath({first, second}, samples, stats).success, "flat discontinuity rejected");

        // full_df enforces reconstructed physical endpoints, independently of lifting.
        Quad2DDF::FlatTrajectory full;
        full.T = 1; full.coefficients(0, 2) = .2; full.coefficients(0, 3) = .01;
        full.start = full.evaluate(0); full.end = full.evaluate(1);
        auto f = fixture(6, 2, 2);
        Quad2DDF::PhysicalState x0, x1;
        require(Quad2DDF::flatToPhysicalState(full.start, full.evaluateControl(0), {}, x0, c2), "full start reconstruction");
        require(Quad2DDF::flatToPhysicalState(full.end, full.evaluateControl(1), {}, x1, c2), "full goal reconstruction");
        f.data->start = x0; f.data->goal = x1;
        Quad2DLift l0{full.acceleration(0), full.jerk(0)}, l1{full.acceleration(1), full.jerk(1)};
        v.df_mode = "full_df";
        DynoSystemModel<Quad2DDF, FixtureAdapter> complete(f, l0, l1, c2, v);
        require(complete.validatePath({full}, samples, stats).success, "full consistent endpoint accepted");
        f.data->goal(2) += .2;
        const auto checked = complete.validatePath({full}, samples, stats);
        require(!checked.success && checked.reason == "full_physical_boundary_mismatch", "full endpoint mismatch rejected");
    });
}
