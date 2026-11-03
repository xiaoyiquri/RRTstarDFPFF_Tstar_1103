#include "core/rrtstar.h"
#include "systems/linear_model.h"
#include "tests/test_support.h"
using namespace rrtpff;
class FullGoalLinearModel : public LinearModel {
  public:
    explicit FullGoalLinearModel(Config c) : LinearModel(c) {
        goal_ = c.start;
        goal_.setDerivative(0, c.goal);
        goal_.setDerivative(1, FlatOutput::Zero());
    }
    TerminalConstraint<2, 2> expansionConstraint(const FlatOutput &output) const {
        return (output - goalOutput()).norm() < 1e-12 ? TerminalConstraint<2, 2>::full(goal_)
                                                      : TerminalConstraint<2, 2>::output(output);
    }
    bool isGoal(const FlatState &state, double tolerance) const {
        return (state.values - goal_.values).norm() <= tolerance;
    }

  private:
    FlatState goal_;
};
int main() {
    return runTest([] {
        LinearModel::Config mc;
        mc.start.values << 1, 1, 0, 0;
        mc.goal << 2, 2;
        mc.upper << 4, 4;
        PlannerConfig pc;
        pc.timeout = 5;
        pc.goal_bias = 1;
        pc.stop_at_first_solution = true;
        pc.max_iterations = 10;
        pc.max_nodes = 10;
        pc.goal_tolerance = 1e-7;
        LinearModel model(mc);
        RRTstar<LinearModel> planner(model, pc);
        const auto result = planner.solve();
        require(result.success && result.node_count == 2, "Direct linear problem solved");
        require(result.dyno_validation_count == 0 && result.simd_eval_count == 0,
                "No false backend metrics");
        double cost = 0, duration = 0;
        PlanResult validation;
        auto previous = model.start();
        for (const auto &tr : planner.path()) {
            require(model.trajectoryValid(tr, validation), "Stored edge revalidation");
            require((tr.start.values - previous.values).norm() < 1e-8, "Path continuity");
            previous = tr.end;
            cost += tr.cost;
            duration += tr.T;
        }
        require(model.isGoal(previous, pc.goal_tolerance), "Path reaches goal");
        for (bool simd : {false, true}) {
            auto full_config = pc;
            full_config.simd_enabled = simd;
            RRTstar<FullGoalLinearModel> full(FullGoalLinearModel(mc), full_config);
            const auto outcome = full.solve();
            require(outcome.success, "full-state goal hook reaches zero-velocity goal");
            require(full.path().back().end.derivative(1).norm() < 1e-7,
                    "goal expansion cannot silently use free terminal velocity");
        }
        close(cost, result.cost, 1e-12, "Reported cost");
        close(duration, result.duration, 1e-12, "Reported duration");
        // A thin obstacle lying between coarse sample locations must be rejected.
        auto obstacle_config = mc;
        LinearModel::Circle circle;
        circle.center << 1.4, 1.4;
        circle.radius = .01;
        obstacle_config.circles.push_back(circle);
        LinearModel blocked(obstacle_config);
        require(!blocked.trajectoryValid(planner.path().front(), validation),
                "Continuous polynomial collision check");
        SIMDCollisionScreen<2, 2>::Sphere thin;
        thin.center = circle.center;
        thin.radius = circle.radius;
        require(SIMDCollisionScreen<2, 2>{}.screen(planner.path().front(), {}, {thin}, 2),
                "Sparse broad phase is not a collision certificate");
        // A sparse pre-screen may miss this thin obstacle; exact checking remains mandatory.
        for (bool enabled : {false, true}) {
            pc.simd_enabled = enabled;
            RRTstar<LinearModel> repeat(model, pc);
            auto run = repeat.solve();
            require(run.success, "Planner runtime backend toggle");
            close(run.cost, result.cost, 1e-7, "Direct solution backend parity");
        }
        LinearModel::Config many;
        many.start.values << 1, 1, 0, 0;
        many.goal << 8, 8;
        many.upper << 10, 10;
        PlannerConfig regression;
        regression.timeout = 30;
        regression.max_iterations = 160;
        regression.max_nodes = 180;
        regression.goal_bias = .2;
        regression.goal_tolerance = .01;
        regression.seed = 45;
        regression.prune = true;
        LinearModel environment(many);
        RRTstar<LinearModel> baseline(environment, regression);
        auto baseline_result = baseline.solve();
        regression.simd_enabled = true;
        RRTstar<LinearModel> accelerated(environment, regression);
        auto accelerated_result = accelerated.solve();
        require(baseline_result.success && accelerated_result.success, "Seeded tree reaches goal");
        require(baseline_result.iterations == regression.max_iterations &&
                    accelerated_result.iterations == regression.max_iterations,
                "Fixed-work benchmark avoids time-budget drift");
        close(accelerated_result.cost, baseline_result.cost, 2e-6, "Seeded scalar/SIMD final cost");
        require(accelerated_result.rewire_count > 0, "Planner exercises rewiring");
        if (simdRuntimeAVX2()) {
            require(accelerated_result.simd_pff_lanes > 0 && accelerated_result.simd_nn_lanes > 0 &&
                        accelerated_result.simd_flat_samples > 0,
                    "All three SIMD hot paths reached");
        }
        for (const auto &node : accelerated.tree().nodes())
            if (node->parent) {
                require((node->state.values - node->incoming.end.values).norm() < 1e-7,
                        "Frozen target state");
                require((node->parent->state.values - node->incoming.start.values).norm() < 1e-7,
                        "Parent edge continuity");
                close(node->cost, node->parent->cost + node->incoming.cost, 1e-10,
                      "Descendant cumulative cost");
                require(environment.trajectoryValid(node->incoming, validation),
                        "Every inserted edge passes exact revalidation");
            }
        // Invalid initial state fails before planning; no approximate success flag.
        mc.start.values(0) = -1;
        RRTstar<LinearModel> invalid(LinearModel(mc), pc);
        require(!invalid.solve().success, "Invalid initial state rejected");
    });
}
