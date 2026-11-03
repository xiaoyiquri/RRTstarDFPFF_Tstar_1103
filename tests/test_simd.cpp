#include "core/nearest_neighbor.h"
#include "core/node.h"
#include "simd/simd_collision.h"
#include "simd/simd_flat_eval.h"
#include "simd/simd_pff.h"
#include "simd/simd_robot_collision.h"
#include "systems/linear_model.h"
#include "tests/test_support.h"
#include <random>

using namespace rrtpff;

template <int Dim, int Order> void checkPFF() {
    using Solver = PFFSolver<Dim, Order>;
    using State = typename Solver::State;
    using Constraint = typename Solver::Constraint;
    typename Solver::Weight R = Solver::Weight::Identity();
    R(0, 0) = 2;
    R(0, 1) = R(1, 0) = .2;
    PFFConfig config;
    config.min_time = .05;
    config.max_time = 6;
    Solver scalar(R, config);
    SIMD_PFF<Dim, Order, std::monostate, 8> batch(scalar);
    std::mt19937 random(40 + Order + Dim);
    std::uniform_real_distribution<double> sample(-.3, .3);
    for (int trial = 0; trial < 12; ++trial)
        for (int mode = 0; mode < 3; ++mode) {
            std::array<State, 8> starts{}, ends{};
            std::array<Constraint, 8> constraints{};
            for (int i = 0; i < 8; ++i) {
                for (int d = 0; d < Dim * Order; ++d) {
                    starts[i].values(d) = sample(random);
                    ends[i].values(d) = sample(random);
                }
                constraints[i] =
                    mode == 1 ? Constraint::full(ends[i]) : Constraint::output(ends[i].output());
                if (mode == 2) {
                    constraints[i].target = ends[i];
                    constraints[i].fixed[Dim] = true;
                }
            }
            const std::size_t count = 1 + trial % 8;
            std::size_t lanes = 0;
            auto accelerated = batch.candidatesBatch(starts, constraints, count, true, &lanes);
            if (!simdRuntimeAVX2() || count < 4)
                require(lanes == 0, "No false SIMD candidate count");
            for (std::size_t i = 0; i < count; ++i) {
                auto reference = scalar.candidates(starts[i], constraints[i]);
                require(accelerated[i].size() == reference.size(),
                        "All duration roots and bounds retained");
                for (std::size_t k = 0; k < reference.size(); ++k) {
                    const auto &a = accelerated[i][k];
                    const auto &b = reference[k];
                    close(a.T, b.T, 2e-7, "Scalar / SIMD time");
                    close(a.cost, b.cost, 2e-7, "Scalar / SIMD cost");
                    require((a.terminal_state.values - b.terminal_state.values).norm() < 2e-5,
                            "Scalar / SIMD terminal state");
                    if (a.T > config.min_time + 1e-6 && a.T < config.max_time - 1e-6)
                        close(scalar.hamiltonian(a), 0, 2e-5, "Interior duration stationary");
                }
            }
            for (std::size_t i = count; i < 8; ++i)
                require(accelerated[i].empty(), "Inactive PFF lane untouched");
        }
    std::array<State, 8> states{};
    std::array<Constraint, 8> constraints{};
    for (auto &c : constraints)
        c = Constraint::output(State::Output::Ones());
    states[3].values(0) = std::numeric_limits<double>::quiet_NaN();
    auto invalid = batch.evaluateBatch(states, constraints);
    require(!invalid.valid[3] && invalid.valid[2], "Bad lane does not poison other candidates");
    constraints[1] = Constraint::full(states[0]);
    auto mixed = batch.candidatesBatch(states, constraints);
    require(!mixed[1].empty(), "Mixed terminal masks safely fall back");
}

template <int Dim, int Order> void checkFlat() {
    using Trajectory = FlatTrajectory<Dim, Order>;
    std::array<Trajectory, 8> storage{};
    std::array<const Trajectory *, 8> edges{};
    std::array<double, 8> times{};
    for (int i = 0; i < 8; ++i) {
        auto &tr = storage[i];
        tr.T = 1.7;
        edges[i] = &tr;
        times[i] = tr.T * i / 7;
        for (int a = 0; a < Dim; ++a)
            for (int k = 0; k <= tr.Degree; ++k)
                tr.coefficients(a, k) = std::sin(i + a + k + .2) * .1;
    }
    SIMDFlatEvaluator<Dim, Order> evaluator;
    for (std::size_t count = 0; count <= 8; ++count) {
        auto input = edges;
        for (std::size_t i = count; i < 8; ++i)
            input[i] = nullptr;
        auto batch = evaluator.evaluateSoA(input, times, count, true);
        for (std::size_t i = 0; i < count; ++i) {
            require((batch.state(i).values - storage[i].evaluateState(times[i]).values).norm() <
                        1e-10,
                    "Horner flat state including tail");
            require((batch.control(i).values - storage[i].evaluateControl(times[i]).values).norm() <
                        1e-10,
                    "Highest derivative is flat control");
        }
    }
    times[0] = -1;
    bool rejected = false;
    try {
        evaluator.evaluateSoA(edges, times, 1);
    } catch (const std::out_of_range &) {
        rejected = true;
    }
    require(rejected, "Out-of-domain time rejected");
}

void checkRoots() {
    std::array<std::vector<long double>, 8> polynomials{};
    polynomials[0] = {-2, 5, -4, 1}; // repeated positive root 1, simple root 2
    polynomials[1] = {0, 0, 1};
    polynomials[2] = {0};
    polynomials[3] = {1, 0, 1};
    polynomials[4] = {-1, 0, 0, 0, 1};
    polynomials[5] = {-1, 1};
    polynomials[6] = {6, -5, 1};
    polynomials[7] = {-24, 50, -35, 10, -1};
    auto roots = batchRoots<8>(polynomials, 0, 5, 8, selectSIMDBackend(true));
    for (int i = 0; i < 8; ++i) {
        auto reference = realRootsInInterval(polynomials[i], 0, 5);
        require(roots[i].size() == reference.size(), "Repeated / endpoint / no root parity");
        for (std::size_t k = 0; k < reference.size(); ++k)
            close(roots[i][k], reference[k], 1e-9, "Bracketed SIMD root");
    }
}

void checkNearest() {
    using Node = RRTNode<2, 2>;
    RRTTree<Node> tree;
    LinearModel model(LinearModel::Config{});
    auto *root = tree.createNode(model.start());
    for (int i = 0; i < 67; ++i) {
        Node::State x;
        x.values << .1 * i, std::sin(i), 0, 0;
        tree.createNode(x, root);
    }
    NearestNeighbor<Node, LinearModel> scalar(tree, model), simd(tree, model, true);
    for (int phase = 0; phase < 3; ++phase) {
        for (int i = 0; i < 37; ++i) {
            Eigen::Vector2d q;
            q << .17 * i, .7;
            require(scalar.nearest(q) == simd.nearest(q), "Nearest exact result and tail");
            std::vector<Node *> a, b;
            scalar.near(q, .8, a);
            simd.near(q, .8, b);
            require(a == b, "Radius results and deterministic order");
            scalar.nearestK(q, 7, a);
            simd.nearestK(q, 7, b);
            require(a == b, "KNN ordering");
        }
        require(tree.removeNode(tree.nodes()[3].get()), "Prune cache fixture");
    }
    tree.clear();
    root = tree.createNode(model.start());
    require(simd.nearest(model.start().output()) == root, "Cache safe after clear and reset");
}

void checkScreen() {
    using Screen = SIMDCollisionScreen<2, 2>;
    FlatTrajectory<2, 2> edge;
    edge.T = 1;
    edge.coefficients(0, 1) = 1;
    Screen screen;
    Screen::Sphere sphere;
    sphere.center << .5, 0;
    sphere.radius = .1;
    for (bool interleaved : {false, true})
        for (std::size_t samples : {2, 7, 8, 9, 17, 33, 65}) {
            bool a = screen.screen(edge, Screen::Limits{}, std::vector<Screen::Sphere>{sphere},
                                   samples, interleaved, false);
            bool b = screen.screen(edge, Screen::Limits{}, std::vector<Screen::Sphere>{sphere},
                                   samples, interleaved, true);
            require(a == b, "Collision screening masks and tails");
        }
    Screen::Limits limits;
    limits.control = .1;
    edge.coefficients(0, 2) = 1;
    require(!screen.screen(edge, limits, {}, 33), "Control-bound sample rejection");
    edge.coefficients(0, 2) = 0;
    edge.coefficients(0, 0) = 1;
    sphere.center << 0, 0;
    sphere.radius = 1;
    require(!screen.screen(edge, Screen::Limits{}, std::vector<Screen::Sphere>{sphere}),
            "Tangent endpoint collision");
}

void checkPaperFloat8Kernels() {
    // The paper path uses eight float samples in one AVX2 register. Compare
    // its lane results with the scalar float fallback, including the tail.
    std::array<float, 16> coefficients{};
    std::array<float, 8> times{};
    for (std::size_t lane = 0; lane < times.size(); ++lane) {
        times[lane] = static_cast<float>(lane) / 7.0f;
        coefficients[lane] = .2f;
        coefficients[8 + lane] = 1.5f;
    }
    std::array<float, 8> scalar{}, vector{};
    polynomialBatchFloat(coefficients.data(), 1, times.data(), times.size(), scalar.data(),
                         SIMDBackend::Scalar);
    polynomialBatchFloat(coefficients.data(), 1, times.data(), times.size(), vector.data(),
                         selectSIMDBackend(true));
    for (std::size_t lane = 0; lane < times.size(); ++lane)
        close(vector[lane], scalar[lane], 2e-6, "Float8 polynomial lane");

    std::array<float, 24> positions{};
    std::array<float, 3> lower{-0.1f, -0.1f, -0.1f};
    std::array<float, 3> upper{0.1f, 0.1f, 0.1f};
    for (std::size_t lane = 0; lane < times.size(); ++lane) {
        positions[lane] = lane == 0 ? 0.0f : 0.2f;
        positions[8 + lane] = 0.0f;
        positions[16 + lane] = 0.0f;
    }
    std::array<unsigned char, 8> scalar_hit{}, vector_hit{};
    sphereAABBBatchFloat(positions.data(), lower.data(), upper.data(), 3, .05f,
                         times.size(), scalar_hit.data(), SIMDBackend::Scalar);
    sphereAABBBatchFloat(positions.data(), lower.data(), upper.data(), 3, .05f,
                         times.size(), vector_hit.data(), selectSIMDBackend(true));
    for (std::size_t lane = 0; lane < times.size(); ++lane)
        require(vector_hit[lane] == scalar_hit[lane], "Float8 collision lane");

    FlatTrajectory<2, 2> edge;
    edge.T = 1.0;
    edge.coefficients(0, 1) = .5;
    CollisionAABB obstacle;
    obstacle.lower = {.4, -.1, 0};
    obstacle.upper = {.6, .1, 0};
    PlanResult stats;
    require(!robotCollisionScreen(edge, 2, .05, {obstacle}, 8, true, stats),
            "Float8 discrete robot collision screen rejects hit");
    if (simdRuntimeAVX2())
        require(stats.simd_collision_samples >= 8, "Float8 collision sample accounting");
}

// Opt-in metric tests are independent of any robot adapter. The periodic axis
// must wrap identically in scalar and SIMD nearest/radius queries.
void checkPeriodicMetric() {
    struct Metric {
        enum { FlatDim = 2 };
        using FlatOutput = Eigen::Vector2d;
        FlatOutput metricWeights() const {
            return {2.0, 0.5};
        }
        FlatOutput metricPeriods() const {
            return {0.0, 6.283185307179586};
        }
        double distanceToOutput(const FlatState<2, 2> &state, const FlatOutput &q) const {
            auto d = (state.output() - q).eval();
            const double p = metricPeriods()(1);
            d(1) -= p * std::floor(d(1) / p + 0.5);
            return std::sqrt((metricWeights().array() * d.array().square()).sum());
        }
    } model;
    using Node = RRTNode<2, 2>;
    RRTTree<Node> tree;
    Node::State zero;
    auto *root = tree.createNode(zero);
    for (int i = 0; i < 18; ++i) {
        Node::State x;
        x.values << i * 0.01, (i % 2 ? -3.12 : 3.12), 0, 0;
        tree.createNode(x, root);
    }
    NearestNeighbor<Node, Metric> scalar(tree, model), vector(tree, model, true);
    for (double angle : {-9.45, -3.15, 3.15, 9.45}) {
        Metric::FlatOutput query(0.045, angle);
        require(scalar.nearest(query) == vector.nearest(query), "Periodic nearest parity");
        std::vector<Node *> a, b;
        scalar.near(query, 0.1, a);
        vector.near(query, 0.1, b);
        require(a == b, "Periodic radius parity");
    }
}

int main() {
    return runTest([] {
        std::array<float, 8> a{1, 2, 3, 4, 5, 6, 7, 8}, b{2, 2, 2, 2, 2, 2, 2, 2}, out{};
        (SimdFloat8::load(a.data()) * SimdFloat8::load(b.data())).store(out.data());
        for (int i = 0; i < 8; ++i)
            close(out[i], 2 * a[i], 0, "Float8 backend arithmetic");
        checkPFF<2, 2>();
        checkPFF<2, 4>();
        checkPFF<4, 4>();
        checkFlat<2, 2>();
        checkFlat<2, 4>();
        checkFlat<4, 4>();
        checkRoots();
        checkNearest();
        checkPeriodicMetric();
        checkScreen();
        checkPaperFloat8Kernels();
        std::cout << "Compiled AVX2=" << simdCompiledAVX2() << " runtime AVX2=" << simdRuntimeAVX2()
                  << '\n';
    });
}
