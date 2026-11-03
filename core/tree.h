#pragma once
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rrtpff {
template <class Node> class RRTTree {
  public:
    Node *root() const {
        return nodes_.empty() ? nullptr : nodes_.front().get();
    }

    Node *createNode(const typename Node::State &state, Node *parent = nullptr,
                     const typename Node::Trajectory &edge = {},
                     const typename Node::MetadataType &root_metadata = {}) {
        if (!nodes_.empty() && !parent)
            throw std::invalid_argument("Only the root may have no parent");
        if (parent && !contains(parent))
            throw std::invalid_argument("Parent does not belong to tree");

        auto node = std::make_unique<Node>();
        node->id = next_id_++;
        node->state = state;
        node->parent = parent;
        node->incoming = edge;
        node->metadata = parent ? edge.metadata : root_metadata;
        node->cost = parent ? parent->cost + edge.cost : 0;

        auto *ptr = node.get();
        nodes_.push_back(std::move(node));
        if (parent)
            parent->addChild(ptr);
        return ptr;
    }

    bool contains(const Node *node) const {
        return std::any_of(nodes_.begin(), nodes_.end(),
                           [node](const auto &p) { return p.get() == node; });
    }

    bool removeNode(Node *node) {
        if (!node || node == root() || !node->children.empty())
            return false;

        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [node](const auto &p) { return p.get() == node; });
        if (it == nodes_.end())
            return false;

        node->parent->removeChild(node);
        nodes_.erase(it);
        ++epoch_;
        return true;
    }

    std::size_t epoch() const {
        return epoch_;
    }

    std::size_t size() const {
        return nodes_.size();
    }

    const auto &nodes() const {
        return nodes_;
    }

    void clear() {
        nodes_.clear();
        next_id_ = 0;
        ++epoch_;
    }

  private:
    std::vector<std::unique_ptr<Node>> nodes_;
    std::size_t next_id_{0};
    std::size_t epoch_{0};
};
} // namespace rrtpff
