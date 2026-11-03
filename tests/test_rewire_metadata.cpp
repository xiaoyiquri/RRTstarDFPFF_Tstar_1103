#include "core/node.h"
#include "core/tree.h"
#include "core/rewiring.h"
#include "pff/pff.h"
#include "systems/unicycle_model.h"
#include "tests/test_support.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

using namespace rrtpff;

using Node = RRTNode<2, 2, UnicycleTrajectoryMetadata>;

bool sameMetadata(const Node::Metadata& a, const Node::Metadata& b) {
    return a.branch == b.branch &&
        a.signed_start_velocity == b.signed_start_velocity;
}

void requireUnchanged(const Node& node, const Node& before) {
    require((node.state.values.array() == before.state.values.array()).all() &&
                sameMetadata(node.metadata, before.metadata) && node.id == before.id,
            "Existing node state and metadata are frozen");
    require(node.parent == before.parent && node.children == before.children &&
                node.cost == before.cost,
            "Rejected rewire leaves tree links and all costs untouched");
    require((node.incoming.coefficients.array() == before.incoming.coefficients.array()).all() &&
                (node.incoming.start.values.array() == before.incoming.start.values.array()).all() &&
                (node.incoming.end.values.array() == before.incoming.end.values.array()).all() &&
                node.incoming.T == before.incoming.T &&
                node.incoming.cost == before.incoming.cost,
            "Stored trajectory stays exactly unchanged");
    if (node.parent) {
        require(sameMetadata(node.incoming.metadata, before.incoming.metadata),
                "Stored edge branch and start speed stay unchanged");
    }
}

void checkBranchPreservation(VelocityBranch branch) {
    UnicycleDF::Config config;
    config.allow_reverse = true;
    config.max_velocity = 2.0;
    const double sign = branch == VelocityBranch::Reverse ? -1.0 : 1.0;

    Node::State root_state, added_state, target_state, child_state, grandchild_state;
    root_state.values << 0.0, 0.0, 0.2, 0.0;
    added_state.values << 0.1, 0.0, 0.2, 0.0;
    target_state.values << 0.325, 0.0, 0.25, 0.0;
    child_state.values << 0.6, 0.0, 0.3, 0.0;
    grandchild_state.values << 0.925, 0.0, 0.35, 0.0;
    UnicycleModel model(root_state, grandchild_state.output(),
                        Node::State::Output::Constant(-2.0),
                        Node::State::Output::Constant(2.0), UniAdapter{},
                        config, sign * 0.2);
    PFFSolver<2, 2, Node::Metadata> solver;
    RRTTree<Node> tree;
    auto edge = [&](const Node::State& from, const Node::State& to, double time) {
        auto result = solver.fixedTime(from, TerminalConstraint<2, 2>::full(to), time);
        require(result.success, "Unicycle full-state PFF edge solved");
        Node::Metadata metadata;
        require(UnicycleDF::makeMetadata(sign * from.derivative(1).norm(),
                                          metadata, config),
                "Explicit start branch");
        require(model.prepareEdge(from, metadata, result.trajectory),
                "Attach start metadata");
        require(model.dfFeasible(result.trajectory), "Fixture DF edge is feasible");
        return result.trajectory;
    };
    auto* root = tree.createNode(root_state, nullptr, {}, model.initialMetadata());
    auto add = [&](const Node::State& state, Node* parent, double time) {
        auto trajectory = edge(parent->state, state, time);
        auto* node = tree.createNode(state, parent, trajectory);
        node->metadata = model.terminalMetadata(trajectory);
        return node;
    };
    auto* added = add(added_state, root, 0.5);
    auto* target = add(target_state, root, 0.5);
    auto* child = add(child_state, target, 1.0);
    auto* grandchild = add(grandchild_state, child, 1.0);
    const auto replacement = edge(added_state, target_state, 1.0);
    require(added->cost + replacement.cost < target->cost,
            "Metadata fixture has a genuine cost improvement");
    require(std::abs(replacement.metadata.signed_start_velocity -
                     target->metadata.signed_start_velocity) > 0.01,
            "Fixture distinguishes edge start speed from endpoint speed");
    require(model.terminalMetadataMatches(replacement, target->metadata),
            "Compatibility uses terminal speed, not the edge start speed");

    // Accept numerical tolerance without overwriting the original metadata.
    target->metadata.signed_start_velocity += sign * 5e-10;
    const std::array<Node*, 5> nodes{root, added, target, child, grandchild};
    const std::array<Node, 5> before{*root, *added, *target, *child, *grandchild};
    auto requireUnchangedTree = [&] {
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            requireUnchanged(*nodes[i], before[i]);
        }
    };

    auto opposite_branch = replacement;
    opposite_branch.metadata.branch = branch == VelocityBranch::Forward
        ? VelocityBranch::Reverse : VelocityBranch::Forward;
    opposite_branch.metadata.signed_start_velocity *= -1.0;
    require(!Rewiring<Node>::commit(target, added, opposite_branch, model),
            "Same flat endpoint with opposite physical branch is rejected");
    requireUnchangedTree();

    auto missing_metadata = replacement;
    missing_metadata.metadata.signed_start_velocity =
        std::numeric_limits<double>::quiet_NaN();
    require(!Rewiring<Node>::commit(target, added, missing_metadata, model),
            "Missing edge metadata is rejected before changing the tree");
    requireUnchangedTree();

    target->metadata.signed_start_velocity = sign * 0.3;
    const Node invalid_target = *target;
    require(!Rewiring<Node>::commit(target, added, replacement, model),
            "Mismatched endpoint signed speed is rejected");
    requireUnchanged(*target, invalid_target);
    target->metadata = before[2].metadata;
    requireUnchangedTree();

    require(Rewiring<Node>::commit(target, added, replacement, model),
            "Same branch and matching endpoint can rewire");
    require(target->parent == added &&
                sameMetadata(target->metadata, before[2].metadata) &&
                (target->state.values.array() == before[2].state.values.array()).all(),
            "Successful rewire preserves the exact target state and metadata");
    require(target->children == before[2].children &&
                sameMetadata(target->incoming.metadata, replacement.metadata),
            "Target keeps children while storing the new edge's start metadata");
    const double delta = target->cost - before[2].cost;
    for (std::size_t i : {std::size_t{3}, std::size_t{4}}) {
        close(nodes[i]->cost, before[i].cost + delta, 1e-12,
              "Only cumulative cost changes in the subtree");
        Node expected = before[i];
        expected.cost = nodes[i]->cost;
        requireUnchanged(*nodes[i], expected);
    }
    require(root->cost == before[0].cost && added->cost == before[1].cost &&
                sameMetadata(added->metadata, before[1].metadata),
            "New parent keeps its state semantics and cost");
}

int main() {
    return runTest([] {
        checkBranchPreservation(VelocityBranch::Forward);
        checkBranchPreservation(VelocityBranch::Reverse);
    });
}
