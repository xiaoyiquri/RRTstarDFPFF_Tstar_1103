#pragma once
#include "core/tree.h"
#include "simd/simd_backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace rrtpff {
template <class M, class = void> struct HasSIMDMetric : std::false_type {};
template <class M>
struct HasSIMDMetric<M, std::void_t<decltype(std::declval<const M &>().metricWeights()),
                                    decltype(std::declval<const M &>().metricPeriods())>>
    : std::true_type {};

// Models opt into a weighted Euclidean/periodic metric. Arbitrary model metrics
// always use the original scalar scan; no robot-specific formulas live here.
template <class Node, class Model> class NearestNeighbor {
  public:
    using Output = typename Model::FlatOutput;
    using Real = typename Output::Scalar;
    NearestNeighbor(const RRTTree<Node> &tree, const Model &model, bool enabled = false)
        : tree_(tree), model_(model), backend_(selectSIMDBackend(enabled)) {
    }
    Node *nearest(const Output &query) const {
        if constexpr (HasSIMDMetric<Model>::value) {
            if (backend_ == SIMDBackend::AVX2 && tree_.size() >= (std::is_same_v<Real,float> ? 8 : 4)) {
                distances(query);
                std::size_t best = 0;
                for (std::size_t i = 1; i < nodes_.size(); ++i) {
                    const Real tol = Real(64) * std::numeric_limits<Real>::epsilon() *
                                       (Real(1) + std::abs(squared_[i]) + std::abs(squared_[best]));
                    if (std::abs(squared_[i] - squared_[best]) <= tol) {
                        if (model_.distanceToOutput(nodes_[i]->state, query) <
                            model_.distanceToOutput(nodes_[best]->state, query))
                            best = i;
                    } else if (squared_[i] < squared_[best])
                        best = i;
                }
                return nodes_.empty() ? nullptr : nodes_[best];
            }
        }
        Node *best = nullptr;
        Real d = std::numeric_limits<Real>::infinity();
        for (const auto &node : tree_.nodes()) {
            const Real next = model_.distanceToOutput(node->state, query);
            if (next < d) {
                best = node.get();
                d = next;
            }
        }
        return best;
    }
    void near(const Output &query, Real radius, std::vector<Node *> &result) const {
        result.clear();
        if (radius < 0 || !std::isfinite(radius))
            return;
        if constexpr (HasSIMDMetric<Model>::value) {
            if (backend_ == SIMDBackend::AVX2 && tree_.size() >= (std::is_same_v<Real,float> ? 8 : 4)) {
                distances(query);
                const Real r2 = radius * radius;
                for (std::size_t i = 0; i < nodes_.size(); ++i) {
                    const Real tol = Real(64) * std::numeric_limits<Real>::epsilon() *
                                       (Real(1) + r2 + std::abs(squared_[i]));
                    if (std::abs(squared_[i] - r2) <= tol) {
                        if (model_.distanceToOutput(nodes_[i]->state, query) <= radius)
                            result.push_back(nodes_[i]);
                    } else if (squared_[i] < r2)
                        result.push_back(nodes_[i]);
                }
                return;
            }
        }
        for (const auto &node : tree_.nodes())
            if (model_.distanceToOutput(node->state, query) <= radius)
                result.push_back(node.get());
    }
    // Optional kNN: cheap partial selection. Scalar refinement keeps close
    // distance ordering deterministic; tree ownership never changes here.
    void nearestK(const Output &query, std::size_t k, std::vector<Node *> &result) const {
        result.clear();
        if constexpr (HasSIMDMetric<Model>::value) {
            if (backend_ == SIMDBackend::AVX2 && tree_.size() >= (std::is_same_v<Real,float> ? 8 : 4)) {
                distances(query);
                std::vector<std::size_t> indices(nodes_.size());
                for (std::size_t i = 0; i < indices.size(); ++i)
                    indices[i] = i;
                k = std::min(k, indices.size());
                // Compute sqrt from the SIMD squared distances once. Model
                // distance is used for near ties before constructing a strict
                // comparator, rather than a non-transitive tolerance compare.
                std::vector<Real> distances_(indices.size());
                for (std::size_t i = 0; i < indices.size(); ++i)
                    distances_[i] = std::sqrt(squared_[i]);
                std::sort(indices.begin(), indices.end(), [&](auto a, auto b) {
                    return squared_[a] == squared_[b] ? nodes_[a]->id < nodes_[b]->id
                                                      : squared_[a] < squared_[b];
                });
                for (std::size_t i = 0; i < indices.size(); ++i) {
                    const auto index = indices[i];
                    bool refine = false;
                    for (auto j : {i ? i - 1 : i, std::min(i + 1, indices.size() - 1)}) {
                        if (j == i)
                            continue;
                        const auto other = indices[j];
                        Real tol = 64 * std::numeric_limits<Real>::epsilon() *
                                     (Real(1) + squared_[index] + squared_[other]);
                        refine |= std::abs(squared_[index] - squared_[other]) <= tol;
                    }
                    if (refine)
                        distances_[index] = model_.distanceToOutput(nodes_[index]->state, query);
                }
                auto less = [&](auto a, auto b) {
                    return distances_[a] == distances_[b] ? nodes_[a]->id < nodes_[b]->id
                                                          : distances_[a] < distances_[b];
                };
                std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), less);
                for (std::size_t i = 0; i < k; ++i)
                    result.push_back(nodes_[indices[i]]);
                return;
            }
        }
        for (const auto &node : tree_.nodes())
            result.push_back(node.get());
        k = std::min(k, result.size());
        auto less = [&](Node *a, Node *b) {
            Real da = model_.distanceToOutput(a->state, query),
                   db = model_.distanceToOutput(b->state, query);
            return da == db ? a->id < b->id : da < db;
        };
        std::partial_sort(result.begin(), result.begin() + k, result.end(), less);
        result.resize(k);
    }
    std::size_t vectorEvaluations() const {
        return vector_evaluations_;
    }

  private:
    void sync() const {
        if (epoch_ != tree_.epoch()) {
            nodes_.clear();
            epoch_ = tree_.epoch();
        }
        const auto n = tree_.size();
        if (n > capacity_) {
            const auto next = std::max(n, std::max(std::size_t{64}, capacity_ * 2));
            std::vector<Real> data(Model::FlatDim * next);
            for (int a = 0; a < Model::FlatDim; ++a)
                for (std::size_t i = 0; i < nodes_.size(); ++i)
                    data[a * next + i] = coordinates_[a * capacity_ + i];
            coordinates_ = std::move(data);
            capacity_ = next;
        }
        for (std::size_t i = nodes_.size(); i < n; ++i) {
            Node *node = tree_.nodes()[i].get();
            nodes_.push_back(node);
            for (int a = 0; a < Model::FlatDim; ++a)
                coordinates_[a * capacity_ + i] = node->state.output()(a);
        }
        squared_.resize(n);
    }
    void distances(const Output &query) const {
        sync();
        const auto weights = model_.metricWeights(), periods = model_.metricPeriods();
        distanceBatchTyped(coordinates_.data(), query.data(), weights.data(), periods.data(),
                      Model::FlatDim, nodes_.size(), capacity_, squared_.data(), backend_);
        constexpr std::size_t W = std::is_same_v<Real,float> ? 8 : 4;
        vector_evaluations_ += (nodes_.size() / W) * W;
    }
    const RRTTree<Node> &tree_;
    const Model &model_;
    SIMDBackend backend_;
    mutable std::size_t epoch_ = std::numeric_limits<std::size_t>::max(), capacity_{0},
                        vector_evaluations_{0};
    mutable std::vector<Node *> nodes_;
    mutable std::vector<Real> coordinates_, squared_;
};
} // namespace rrtpff
