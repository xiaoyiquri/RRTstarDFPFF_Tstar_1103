#include "core/rrtstar.h"
#include "df/quad3d_df.h"
#include "simd/simd_robot_collision.h"
#include "systems/linear_model.h"
#include "systems/unicycle_model.h"
#include "tests/test_support.h"
#include <iomanip>
#include <random>
#include <sstream>
using namespace rrtpff;

template <int Dim, int Order> void checkDeferred() {
    using S = PFFSolver<Dim, Order>;
    typename S::Weight weight = S::Weight::Identity();
    weight(0, 1) = weight(1, 0) = .13;
    PFFConfig c;
    c.min_time = .3;
    c.max_time = 6;
    c.additional_time_samples = 12;
    S eager(weight, c);
    c.defer_trajectories = true;
    S lazy(weight, c);
    SIMD_PFF<Dim, Order> simd(lazy);
    std::mt19937 rng(41);
    std::uniform_real_distribution<double> dist(-.2, .2);
    for (int mode = 0; mode < 3; ++mode) {
        std::array<typename S::State, SIMDBatchWidth> starts{}, ends{};
        std::array<typename S::Constraint, SIMDBatchWidth> targets{};
        for (std::size_t i = 0; i < SIMDBatchWidth; ++i) {
            for (int d = 0; d < Dim * Order; ++d) {
                starts[i].values(d) = dist(rng);
                ends[i].values(d) = dist(rng);
            }
            targets[i] =
                mode == 0 ? S::Constraint::output(ends[i].output()) : S::Constraint::full(ends[i]);
            if (mode == 2)
                targets[i].fixed[Dim] = false;
        }
        for (std::size_t count = 1; count <= SIMDBatchWidth; ++count) {
            auto batches = simd.candidatesBatch(starts, targets, count, true);
            for (std::size_t i = 0; i < count; ++i) {
                auto ref = eager.candidates(starts[i], targets[i]);
                auto deferred = lazy.candidates(starts[i], targets[i]);
                require(deferred.size() == ref.size(), "eager/lazy retains duration candidates");
                require(batches[i].size() == ref.size(), "k=8 retains duration candidates");
                for (std::size_t k = 0; k < ref.size(); ++k) {
                    require(deferred[k].deferred && deferred[k].polynomial_coefficients.empty(),
                            "unselected curves not materialized");
                    close(deferred[k].cost, ref[k].cost, 1e-12,
                          "same reference cost before materialization");
                    require(lazy.materialize(starts[i], deferred[k]), "lazy curve reconstruction");
                    require(simd.materialize(starts[i], batches[i][k]),
                            "SIMD lazy curve reconstruction");
                    close(deferred[k].T, ref[k].T, 1e-12, "same lazy duration");
                    close(batches[i][k].T, ref[k].T, 2e-7, "k=8 duration parity");
                    close(batches[i][k].cost, ref[k].cost, 2e-7, "k=8 cost parity");
                    require((deferred[k].trajectory.coefficients - ref[k].trajectory.coefficients)
                                    .norm() < 1e-9,
                            "identical materialized polynomials");
                    for (int row = 0; row < Dim * Order; ++row)
                        if (targets[i].fixed[row])
                            close(deferred[k].terminal_state.values(row),
                                  targets[i].target.values(row), 1e-7,
                                  "selected terminal coordinates remain fixed");
                }
            }
        }
    }
}

class GoalModel : public LinearModel {
  public:
    explicit GoalModel(Config c) : LinearModel(c) { goal_.setDerivative(0, c.goal); }
    TerminalConstraint<2, 2> goalConstraint() const {
        return TerminalConstraint<2, 2>::full(goal_);
    }
    TerminalConstraint<2, 2> expansionConstraint(const FlatOutput &y) const {
        return (y - goal_.output()).norm() < 1e-10 ? goalConstraint()
                                                   : TerminalConstraint<2, 2>::output(y);
    }
    bool isGoal(const FlatState &s, double tol) const {
        return (s.values - goal_.values).norm() <= tol;
    }
    bool precheckTrajectory(const FlatTrajectory &tr, PlanResult &) const { return tr.T >= .35; }
    bool trajectoryValidAfterPrecheck(const FlatTrajectory &tr, PlanResult &stats) const {
        return trajectoryValid(tr, stats);
    }
    std::vector<double> additionalCandidateTimes(const FlatState &,
                                                 const TerminalConstraint<2, 2> &) const {
        return {1.234, 2.345};
    }

  private:
    FlatState goal_;
};
std::string treeSnapshot(bool deferred, bool vectorized, bool cache, PlanResult &result) {
    LinearModel::Config mc;
    mc.start.values << 1, 1, 0, 0;
    mc.goal << 3, 3;
    mc.upper << 5, 5;
    GoalModel model(mc);
    PlannerConfig pc;
    pc.max_iterations = 160;
    pc.timeout = 60;
    pc.goal_bias = .2;
    pc.range = .8;
    pc.rewire_factor = 4;
    pc.simd_enabled = vectorized;
    pc.explicit_goal_connection = true;
    pc.cache_goal_connections = cache;
    pc.candidate_precheck = cache;
    pc.model_time_hints = true;
    pc.seed = 41;
    PFFConfig config;
    config.min_time = .1;
    config.max_time = 5;
    config.additional_time_samples = 12;
    config.defer_trajectories = deferred;
    RRTstar<GoalModel> planner(model, pc, PFFSolver<2, 2>(Eigen::Matrix2d::Identity(), config));
    result = planner.solve();
    require(result.iterations == pc.max_iterations, "deterministic fixed iteration run");
    require(result.success && result.rewire_count > 0, "exercise solutions and full-state rewire");
    std::ostringstream out;
    out << std::setprecision(17);
    for (const auto &node : planner.tree().nodes()) {
        out << node->id << ' ' << (node->parent ? static_cast<long long>(node->parent->id) : -1)
            << ' ' << node->cost;
        for (auto value : node->state.values)
            out << ' ' << value;
        for (Eigen::Index i = 0; i < node->incoming.coefficients.size(); ++i)
            out << ' ' << node->incoming.coefficients.data()[i];
        out << '\n';
    }
    return out.str();
}
void checkTree() {
    for (bool vectorized : {false, true})
        for (bool cache : {false, true}) {
            PlanResult a, b;
            const auto eager = treeSnapshot(false, vectorized, cache, a);
            const auto lazy = treeSnapshot(true, vectorized, cache, b);
            require(eager == lazy,
                    "eager/lazy same states, parents, costs and fixed-state rewire edges");
            require(b.pff_materializations > 0 && b.pff_materializations < b.pff_eval_count,
                    "only demanded duration candidates are materialized");
        }
}
void checkCollision() {
    std::mt19937 rng(20);
    std::uniform_real_distribution<double> dist(-2, 2);
    const double lower[3] = {-.4, -.5, -.6}, upper[3] = {.4, .5, .6};
    for (int dim : {2, 3})
        for (std::size_t count = 0; count <= 17; ++count) {
            std::vector<double> x(dim * count);
            for (auto &v : x)
                v = dist(rng);
            if (count > 4)
                x[4] = std::numeric_limits<double>::quiet_NaN();
            std::vector<unsigned char> a(count), b(count);
            sphereAABBBatch(x.data(), lower, upper, dim, .25, count, a.data(), SIMDBackend::Scalar);
            sphereAABBBatch(x.data(), lower, upper, dim, .25, count, b.data(),
                            selectSIMDBackend(true));
            require(a == b, "sphere/AABB scalar-SIMD parity including tails and NaN");
        }
    FlatTrajectory<2, 2> edge;
    edge.T = 1;
    edge.coefficients(0, 1) = 1;
    CollisionAABB obstacle;
    obstacle.lower = {.4, -.2, 0};
    obstacle.upper = {.6, .2, 0};
    PlanResult stats;
    require(!robotCollisionScreen(edge, 2, .1, {obstacle}, 16, true, stats),
            "interior collision rejected");
    obstacle.lower[1] = .2;
    obstacle.upper[1] = .4;
    require(robotCollisionScreen(edge, 2, .1, {obstacle}, 16, true, stats),
            "miss must continue to FCL");
    require(robotCollisionScreen(edge, 2, 0, {obstacle}, 16, true, stats),
            "unknown body screen disabled");
}
void checkChangedStart() {
    PFFConfig config;
    config.defer_trajectories = true;
    PFFSolver<2, 2> solver(Eigen::Matrix2d::Identity(), config);
    FlatState<2, 2> start;
    auto constraint = TerminalConstraint<2, 2>::output(Eigen::Vector2d(1, 2));
    auto candidates = solver.resultsAtTimes(start, constraint, {1.0}, false);
    start.values(0) = .2;
    const auto reference = solver.fixedTime(start, constraint, 1.0);
    require(solver.materialize(start, candidates.front()),
            "changed start forces fresh terminal solve");
    require(
        (reference.trajectory.coefficients - candidates.front().trajectory.coefficients).norm() <
            1e-12,
        "cached terminal solution never reused for different initial state");
}
void checkReverseMetadata() {
    using Solver = PFFSolver<2, 2, UnicycleTrajectoryMetadata>;
    Solver::State start, end;
    start.values << 0, 0, .2, 0;
    end.values << .2, 0, .2, 0;
    UnicycleDF::Config c;
    c.allow_reverse = true;
    UnicycleModel model(start, end.output(), {-2, -2}, {2, 2}, UniAdapter{}, c, -.2);
    PFFConfig config;
    config.defer_trajectories = true;
    Solver solver(Eigen::Matrix2d::Identity(), config);
    auto edges = solver.resultsAtTimes(start, Solver::Constraint::full(end), {1.0}, false);
    require(edges.size() == 1 && edges[0].deferred, "reverse edge deferred");
    require(solver.materialize(start, edges[0]) &&
                model.prepareEdge(start, model.initialMetadata(), edges[0].trajectory),
            "reverse metadata attached after materialization");
    const auto copy = edges[0].trajectory;
    require(copy.metadata.branch == VelocityBranch::Reverse &&
                copy.metadata.signed_start_velocity < 0,
            "copy/cache retains reverse metadata");
    UnicycleDF::PhysicalState x;
    UnicycleDF::PhysicalControl u;
    require(UnicycleDF::trajectoryState(copy, .5, x, c) &&
                UnicycleDF::trajectoryControl(copy, .5, u, c),
            "reverse trajectory reconstruction");
    require(u(0) < 0, "inverse retains negative speed");
    close(u(0) * std::cos(x(2)), .2, 1e-10, "reverse physical velocity equals flat motion");
}
void checkWrench() {
    Quad3DDF::Config c;
    c.mass = 1;
    c.inertia = Eigen::Vector3d(.1, .1, .2);
    c.max_motor = 1.5;
    c.max_acceleration = c.max_angular_acceleration = std::numeric_limits<double>::infinity();
    auto mapping = std::make_shared<WrenchControlMap>();
    mapping->inverse = Eigen::Vector4d(1 / c.gravity, .05, .05, .05).asDiagonal();
    mapping->lower << 0, -.1, -.1, -.1;
    mapping->upper << 1.5, .1, .1, .1;
    c.control_map = mapping;
    Quad3DDF::FlatState flat;
    Quad3DDF::FlatControl snap;
    Quad3DDF::PhysicalState state, separate;
    Quad3DDF::PhysicalControl u, separate_u;
    require(Quad3DDF::sampleFeasible(flat, snap, {}, state, u, c), "paper hover feasible");
    require((u - Eigen::Vector4d(1, 0, 0, 0)).norm() < 1e-12, "hover is total normalized thrust 1");
    for (double yaw_alpha : {-9.9, 9.9}) {
        auto a = flat.derivative(2);
        a(3) = yaw_alpha;
        flat.setDerivative(2, a);
        require(Quad3DDF::sampleFeasible(flat, snap, {}, state, u, c),
                "yaw torque below 2 feasible");
        close(u(3) * 20, .2 * yaw_alpha, 1e-10, "tau=J*alpha in true control units");
        require(Quad3DDF::flatToPhysicalState(flat, snap, {}, separate, c) &&
                    Quad3DDF::flatToPhysicalControl(flat, snap, {}, separate_u, c),
                "separate DF interfaces");
        require((state - separate).norm() < 1e-12 && (u - separate_u).norm() < 1e-12,
                "reuse frame equals reference interfaces");
    }
    auto a = flat.derivative(2);
    a(3) = 10.1;
    flat.setDerivative(2, a);
    require(!Quad3DDF::sampleFeasible(flat, snap, {}, state, u, c), "yaw torque above 2 rejected");
    a.setZero();
    a(2) = .51 * c.gravity;
    flat.setDerivative(2, a);
    require(!Quad3DDF::sampleFeasible(flat, snap, {}, state, u, c), "thrust above 1.5 mg rejected");
}
int main() {
    return runTest([] {
        static_assert(SIMDBatchWidth == 8);
        checkDeferred<2, 2>();
        checkDeferred<2, 4>();
        checkDeferred<4, 4>();
        checkTree();
        checkCollision();
        checkWrench();
        checkReverseMetadata();
        checkChangedStart();
    });
}
