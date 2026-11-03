#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace rrtpff {
// Local solving and validation happen before commit. This class only changes
// one incoming edge, tree ownership links and cumulative costs. Existing node
// states/metadata and all descendant edges remain unchanged.
template <class Node>
class Rewiring {
public:
    using Real = decltype(Node::cost);
    static bool isAncestor(const Node* ancestor, const Node* node) {
        for (auto* p = node; p; p = p->parent)
            if (p == ancestor) return true;
        return false;
    }

    static Real propagateCostToChildren(Node* node) {
        Real largest_change = 0;
        std::vector<Node*> pending{node};
        while (!pending.empty()) {
            Node* parent = pending.back();
            pending.pop_back();
            for (Node* child : parent->children) {
                const Real cost = parent->cost + child->incoming.cost;
                largest_change = std::max(largest_change, std::abs(child->cost - cost));
                child->cost = cost;
                pending.push_back(child);
            }
        }
        return largest_change;
    }

    static bool commit(Node* node, Node* parent,
                       const typename Node::Trajectory& edge) {
        static_assert(std::is_same_v<typename Node::MetadataType, std::monostate>,
                      "Stateful metadata requires a model for rewire validation");
        return commit(node, parent, edge, std::monostate{});
    }

    template <class Model>
    static bool commit(Node* node, Node* parent,
                       const typename Node::Trajectory& edge,
                       const Model& model) {
        if (!node || !parent || !node->parent || isAncestor(node, parent)) return false;
        if (!std::isfinite(edge.cost) || edge.cost <= 0) return false;
        if (!edge.start.values.isApprox(parent->state.values, Real(1e-5)) ||
            !edge.end.values.isApprox(node->state.values, Real(1e-5)))
            return false;
        if (parent->cost + edge.cost >= node->cost - Real(1e-7)) return false;

        // Matching flat coordinates alone need not identify the same physical
        // state. Ask the model to match the endpoint semantics before mutation.
        if constexpr (!std::is_same_v<typename Node::MetadataType, std::monostate>) {
            if (!model.terminalMetadataMatches(edge, node->metadata)) return false;
        }

        node->parent->removeChild(node);
        node->parent = parent;
        parent->addChild(node);
        node->incoming = edge;
        // The target state and its metadata are immutable under rewiring.
        node->cost = parent->cost + edge.cost;
        propagateCostToChildren(node);
        return true;
    }
};
}  // namespace rrtpff
