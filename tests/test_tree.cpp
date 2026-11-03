#include "core/node.h"
#include "core/tree.h"
#include "core/rewiring.h"
#include "pff/pff.h"
#include "tests/test_support.h"
#include <algorithm>
#include <array>
#include <vector>

using namespace rrtpff;

template <int Dim, int Order>
void checkFixedStateRewire() {
    using Node = RRTNode<Dim, Order>;
    using State = typename Node::State;
    using Output = typename State::Output;
    RRTTree<Node> tree;
    PFFSolver<Dim, Order> solver;

    State root_state;
    for (int derivative = 1; derivative < Order; ++derivative) {
        root_state.setDerivative(derivative, Output::Constant(0.05));
    }
    auto a_state = root_state;
    auto b_state = root_state;
    auto c_state = root_state;
    auto d_state = root_state;
    auto e_state = root_state;
    a_state.values(0) = 2.0;
    b_state.values(0) = 1.0;
    c_state.values(0) = 3.0;
    d_state.values(0) = 4.0;
    e_state.values(0) = 3.0;
    e_state.values(1) = 0.5;

    auto edge = [&](const State& from, const State& to, double time) {
        auto result = solver.fixedTime(
            from, TerminalConstraint<Dim, Order>::full(to), time);
        require(result.success, "Fixed full-state test edge solved");
        return result.trajectory;
    };
    auto* root = tree.createNode(root_state);
    auto* a = tree.createNode(a_state, root, edge(root_state, a_state, 0.5));
    auto* b = tree.createNode(b_state, root, edge(root_state, b_state, 1.0));
    auto* c = tree.createNode(c_state, a, edge(a_state, c_state, 1.0));
    auto* d = tree.createNode(d_state, c, edge(c_state, d_state, 1.0));
    auto* e = tree.createNode(e_state, a, edge(a_state, e_state, 1.2));

    const std::array<Node*, 6> nodes{root, a, b, c, d, e};
    const std::array<Node, 6> before{*root, *a, *b, *c, *d, *e};
    const auto replacement = edge(b_state, a_state, 1.5);
    require(b->cost + replacement.cost < a->cost,
            "Fixture provides an actual cost improvement");

    // Every endpoint derivative is fixed, including acceleration/jerk for
    // fourth-order systems. None may be changed to obtain a cheaper edge.
    for (int derivative = 1; derivative < Order; ++derivative) {
        auto invalid = replacement;
        invalid.end.values(derivative * Dim) += 0.1;
        require(!Rewiring<Node>::commit(a, b, invalid),
                "Changed terminal derivative rejected");
        require(a->parent == root && a->cost == before[1].cost &&
                    d->cost == before[4].cost,
                "Rejected endpoint leaves links and costs untouched");
    }

    require(Rewiring<Node>::commit(a, b, replacement),
            "Valid cost-improving rewire");
    const double delta = a->cost - before[1].cost;
    require(delta < 0.0, "Cumulative cost decreases");
    require(a->parent == b && b->children == std::vector<Node*>{a},
            "Only rewired node gets the new parent");
    require(root->children == std::vector<Node*>{b},
            "Old parent removes only the rewired child");
    require(a->children == before[1].children,
            "Rewired node keeps its children");
    require((a->incoming.coefficients.array() == replacement.coefficients.array()).all() &&
                a->incoming.T == replacement.T &&
                a->incoming.cost == replacement.cost,
            "Only the requested incoming trajectory is replaced");

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const Node& current = *nodes[i];
        const Node& original = before[i];
        require((current.state.values.array() == original.state.values.array()).all() &&
                    current.metadata == original.metadata && current.id == original.id,
                "Every existing node keeps its exact state and metadata");
        if (nodes[i] != a) {
            require(current.parent == original.parent,
                    "Other parent pointers remain unchanged");
            require((current.incoming.coefficients.array() ==
                     original.incoming.coefficients.array()).all() &&
                        (current.incoming.start.values.array() ==
                         original.incoming.start.values.array()).all() &&
                        (current.incoming.end.values.array() ==
                         original.incoming.end.values.array()).all() &&
                        current.incoming.T == original.incoming.T &&
                        current.incoming.cost == original.incoming.cost &&
                        current.incoming.metadata == original.incoming.metadata,
                    "All other stored trajectories remain exactly unchanged");
        }
        if (i >= 3) {
            require(current.children == original.children,
                    "Subtree topology remains unchanged");
            close(current.cost, original.cost + delta, 1e-12,
                  "Children and grandchildren inherit only the cost delta");
        }
    }
    require(root->cost == before[0].cost && b->cost == before[2].cost,
            "Costs outside the rewired subtree stay unchanged");
    require(!Rewiring<Node>::commit(a, d, edge(d_state, a_state, 1.0)),
            "Cycle rejected");
    require(!tree.removeNode(b), "Cannot delete internal node");
    require(tree.removeNode(d) && c->children.empty(),
            "Leaf deletion updates parent");
    require(tree.size() == 5, "Tree size");
}

int main() {
    return runTest([] {
        checkFixedStateRewire<2, 2>();
        checkFixedStateRewire<2, 4>();
        checkFixedStateRewire<4, 4>();
    });
}
