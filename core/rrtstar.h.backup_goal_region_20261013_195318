#pragma once
#include "common/planner_config.h"
#include "common/timer.h"
#include "core/nearest_neighbor.h"
#include "core/node.h"
#include "core/rewiring.h"
#include "core/sampler.h"
#include "core/tree.h"
#include "pff/pff.h"
#include "simd/simd_pff.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace rrtpff {
template <class M, class = void> struct HasCandidateTimes : std::false_type {};
template <class M>
struct HasCandidateTimes<
    M, std::void_t<decltype(std::declval<const M &>().additionalCandidateTimes(
           std::declval<const typename M::FlatState &>(),
           std::declval<const TerminalConstraint<M::FlatDim, M::FlatOrder> &>()))>>
    : std::true_type {};
template <class M, class = void> struct HasCandidatePrecheck : std::false_type {};
template <class M>
struct HasCandidatePrecheck<
    M, std::void_t<
           decltype(std::declval<const M &>().precheckTrajectory(
               std::declval<const typename M::FlatTrajectory &>(), std::declval<PlanResult &>())),
           decltype(std::declval<const M &>().trajectoryValidAfterPrecheck(
               std::declval<const typename M::FlatTrajectory &>(), std::declval<PlanResult &>()))>>
    : std::true_type {};
template <class M, class = void> struct HasAdditionalConstraints : std::false_type {};
template <class M>
struct HasAdditionalConstraints<
    M,
    std::void_t<decltype(std::declval<const M &>().additionalExpansionConstraints(
        std::declval<const typename M::FlatState &>(), std::declval<const typename M::Metadata &>(),
        std::declval<const typename M::FlatOutput &>()))>> : std::true_type {};
template <class M, class = void> struct HasExpansionConstraint : std::false_type {};
template <class M>
struct HasExpansionConstraint<M, std::void_t<decltype(std::declval<const M &>().expansionConstraint(
                                     std::declval<const typename M::FlatOutput &>()))>>
    : std::true_type {};
template <class M, class = void> struct HasTrajectoryScreen : std::false_type {};
template <class M, class = void> struct HasGoalConstraint : std::false_type {};
template <class M>
struct HasGoalConstraint<M, std::void_t<decltype(std::declval<const M &>().goalConstraint())>>
    : std::true_type {};
template <class M, class = void> struct HasAlternativeConstraint : std::false_type {};
template <class M>
struct HasAlternativeConstraint<
    M,
    std::void_t<decltype(std::declval<const M &>().alternativeExpansionConstraint(
        std::declval<const typename M::FlatState &>(), std::declval<const typename M::Metadata &>(),
        std::declval<const typename M::FlatOutput &>()))>> : std::true_type {};
template <class M>
struct HasTrajectoryScreen<M, std::void_t<decltype(std::declval<const M &>().screenTrajectory(
                                  std::declval<const typename M::FlatTrajectory &>(), std::size_t{},
                                  bool{}, std::declval<PlanResult &>()))>> : std::true_type {};
template<class M, class = void> struct PlannerSolverTypes {
    using Solver = PFFSolver<M::FlatDim,M::FlatOrder,typename M::Metadata>;
    using BatchSolver = SIMD_PFF<M::FlatDim,M::FlatOrder,typename M::Metadata,SIMDBatchWidth>;
};
template<class M> struct PlannerSolverTypes<M,std::void_t<typename M::Solver,typename M::BatchSolver>> {
    using Solver = typename M::Solver;
    using BatchSolver = typename M::BatchSolver;
};
template <class SystemModelT> class RRTstar {
  public:
    using Model = SystemModelT;
    using State = typename Model::FlatState;
    using Output = typename Model::FlatOutput;
    using Metadata = typename Model::Metadata;
    using Real = typename Output::Scalar;
    using Node = RRTNode<Model::FlatDim, Model::FlatOrder, Metadata, Real>;
    using Solver = typename PlannerSolverTypes<Model>::Solver;
    using Constraint = typename Solver::Constraint;
    using Edge = typename Solver::Result;
    using Trajectory = typename Node::Trajectory;
    static constexpr std::size_t BatchWidth = SIMDBatchWidth;
    using BatchSolver = typename PlannerSolverTypes<Model>::BatchSolver;

    RRTstar(Model model, PlannerConfigT<Real> config, Solver pff = Solver{})
        : model_(std::move(model)), config_(config),
          timing_pff_(config.model_time_hints ? std::optional<Solver>(pff) : std::nullopt),
          batch_pff_(std::move(pff)) {
        config_.validate();
    }

    PlanResult solve() {
        tree_.clear();
        goals_.clear();
        path_.clear();
        first_path_.clear();
        goal_cache_.clear();
        stats_ = {};
        stats_.simd_active = config_.simd_enabled && simdRuntimeAVX2();

        Timer wall;

        if (!model_.initialStateValid() || !model_.goalOutputValid()) {
            stats_.failure_reason = "invalid_endpoint";
            stats_.planning_time = wall.seconds();
            return stats_;
        }

        Node *root = tree_.createNode(model_.start(), nullptr, {}, model_.initialMetadata());
        if (model_.isGoal(root->state, config_.goal_tolerance)) {
            goals_.push_back(root);
            stats_.first_solution_time = wall.seconds();
        }

        Sampler<Model> sampler(model_, config_.seed);
        NearestNeighbor<Node, Model> nn(tree_, model_, config_.simd_enabled && config_.simd_nn);
        std::vector<Node *> parents;
        std::vector<Candidate> candidates;

        while (wall.seconds() < config_.timeout &&
               (!config_.max_iterations || stats_.iterations < config_.max_iterations) &&
               (!config_.max_nodes || tree_.size() < config_.max_nodes)) {
            if (config_.stop_at_first_solution && !goals_.empty())
                break;

            if (config_.hard_safety_node_cap && tree_.size() >= config_.hard_safety_node_cap) {
                stats_.termination_reason = "safety_limit";
                stats_.completed_normally = false;
                break;
            }
            ++stats_.iterations;
            bool sampled_goal = false;
            const Output target = sampler.sampleFlatOutput(config_.goal_bias, &sampled_goal);
            const bool goal_connection =
                sampled_goal && config_.explicit_goal_connection && HasGoalConstraint<Model>::value;

            // Select the nearest parent to the sample before limiting extension.
            Timer nn_timer;
            Node *nearest = nn.nearest(target);
            const Output limited =
                goal_connection ? target
                                : model_.limitExtension(nearest->state, target, config_.range);

            parents.clear();
            const Real n = static_cast<Real>(tree_.size() + 1);
            const Real radius = config_.rewire_factor * config_.range *
                                  std::pow(std::log(n) / n, Real(1) / Real(Model::FlatDim));
            nn.near(limited, radius, parents);
            if (std::find(parents.begin(), parents.end(), nearest) == parents.end())
                parents.push_back(nearest);
            if (!goal_connection && parents.size() < config_.minimum_parent_candidates) {
                std::vector<Node *> extra;
                nn.nearestK(limited, config_.minimum_parent_candidates, extra);
                for (Node *node : extra)
                    if (std::find(parents.begin(), parents.end(), node) == parents.end())
                        parents.push_back(node);
            }
            if (goal_connection) {
                // Bounded goal-parent set: consider distinct full states, not
                // just repeatedly trying the one closest position.
                parents.clear();
                for (const auto &node : tree_.nodes()) {
                    if (config_.cache_goal_connections &&
                        (goalComplete(node.get()) ||
                         model_.isGoal(node->state, config_.goal_tolerance)))
                        continue;
                    parents.push_back(node.get());
                }
                const auto keep = std::min(config_.goal_parent_limit, parents.size());
                std::partial_sort(parents.begin(), parents.begin() + keep, parents.end(),
                                  [&](Node *a, Node *b) {
                                      const Real da = model_.distanceToOutput(a->state, target),
                                                   db = model_.distanceToOutput(b->state, target);
                                      return da == db ? a->id < b->id : da < db;
                                  });
                parents.resize(keep);
                stats_.goal_connection_queries += parents.size();
            }

            stats_.nn_time += nn_timer.seconds();
            candidates.clear();
            if (goal_connection && config_.cache_goal_connections)
                appendCachedGoals(parents, candidates, wall);
            else
                for (std::size_t offset = 0; offset < parents.size(); offset += BatchWidth) {
                    if (wall.seconds() >= config_.timeout)
                        break;
                    const auto count = std::min(BatchWidth, parents.size() - offset);
                    std::array<Node *, BatchWidth> block{};
                    std::array<Constraint, BatchWidth> targets{};
                    for (std::size_t i = 0; i < count; ++i) {
                        block[i] = parents[offset + i];
                        if constexpr (HasExpansionConstraint<Model>::value)
                            targets[i] = model_.expansionConstraint(limited);
                        else
                            targets[i] = Constraint::output(limited);
                        if constexpr (HasGoalConstraint<Model>::value)
                            if (goal_connection)
                                targets[i] = model_.goalConstraint();
                    }
                    appendBatch(block, targets, count, candidates);
                    if constexpr (HasAlternativeConstraint<Model>::value) {
                        if (!goal_connection) {
                            std::array<Node *, BatchWidth> alternatives{};
                            std::array<Constraint, BatchWidth> alternative_targets{};
                            std::size_t used = 0;
                            for (std::size_t i = 0; i < count; ++i) {
                                auto alternative = model_.alternativeExpansionConstraint(
                                    block[i]->state, block[i]->metadata, limited);
                                if (alternative) {
                                    alternatives[used] = block[i];
                                    alternative_targets[used++] = *alternative;
                                }
                            }
                            if (used)
                                appendBatch(alternatives, alternative_targets, used, candidates);
                        }
                    }
                    if constexpr (HasAdditionalConstraints<Model>::value) {
                        if (!goal_connection) {
                            std::array<Node *, BatchWidth> alternatives{};
                            std::array<Constraint, BatchWidth> alternative_targets{};
                            std::size_t used = 0;
                            for (std::size_t i = 0; i < count; ++i) {
                                for (const auto &constraint : model_.additionalExpansionConstraints(
                                         block[i]->state, block[i]->metadata, limited)) {
                                    alternatives[used] = block[i];
                                    alternative_targets[used++] = constraint;
                                    if (used == BatchWidth) {
                                        appendBatch(alternatives, alternative_targets, used,
                                                    candidates);
                                        used = 0;
                                    }
                                }
                            }
                            if (used)
                                appendBatch(alternatives, alternative_targets, used, candidates);
                        }
                    }
                }
            sortCandidates(candidates);

            Node *added = nullptr;
            const bool precheck = config_.candidate_precheck && HasCandidatePrecheck<Model>::value;
            const std::size_t count = candidates.size();
            for (std::size_t i = 0; i < count && wall.seconds() < config_.timeout; ++i) {
                auto &candidate = candidates[i];
                if (!materializeEdge(candidate.edge, candidate.parent)) {
                    rejectCached(candidate);
                    continue;
                }
                if (candidate.cached)
                    candidate.cached->edge = candidate.edge;
                if (precheck && !precheckEdge(candidate.edge.trajectory)) {
                    rejectCached(candidate);
                    continue;
                }
                if (!validateEdge(candidate.edge.trajectory, precheck)) {
                    rejectCached(candidate);
                    continue;
                }
                if (candidate.cached) {
                    auto &entry = goal_cache_.at(candidate.parent->id);
                    entry.connected = true;
                    entry.connected_parent_cost = candidate.parent->cost;
                }
                if (goal_connection)
                    ++stats_.goal_connection_valid;
                added = tree_.createNode(candidate.edge.terminal_state, candidate.parent,
                                         candidate.edge.trajectory);
                added->metadata = model_.terminalMetadata(candidate.edge.trajectory);
                break;
            }
            if (!added)
                continue;

            if (model_.isGoal(added->state, config_.goal_tolerance)) {
                goals_.push_back(added);
                if (!std::isfinite(stats_.first_solution_time)) {
                    stats_.first_solution_time = wall.seconds();
                    stats_.first_solution_cost = double(added->cost);
                    stats_.best_cost_events.emplace_back(stats_.first_solution_time,double(added->cost));
                    for (Node *n=added;n->parent;n=n->parent) first_path_.push_back(n->incoming);
                    std::reverse(first_path_.begin(),first_path_.end());
                }
                if (config_.stop_at_first_solution) {
                    stats_.termination_reason = "first_solution";
                    break;
                }
            }

            //if (config_.rewire_before_first_solution || !goals_.empty())
            //    rewire(added, parents, wall);
            
            if (config_.enable_rewire &&(config_.rewire_before_first_solution || !goals_.empty())) {
    		rewire(added, parents, wall);
	    }

            if (config_.prune)
                pruneTree();
            if (Node *g=bestGoal()) {
                if (stats_.best_cost_events.empty() || double(g->cost) < stats_.best_cost_events.back().second)
                    stats_.best_cost_events.emplace_back(wall.seconds(),double(g->cost));
            }
        }

        if (stats_.termination_reason.empty()) {
            if (config_.max_iterations && stats_.iterations >= config_.max_iterations)
                stats_.termination_reason = "max_iterations";
            else if (config_.max_nodes && tree_.size() >= config_.max_nodes)
                stats_.termination_reason = "max_nodes";
            else stats_.termination_reason = "timeout";
        }
        Node *best = bestGoal();
        stats_.node_count = tree_.size();
        stats_.simd_nn_lanes = nn.vectorEvaluations();
        stats_.simd_eval_count =
            stats_.simd_pff_lanes + stats_.simd_flat_samples + stats_.simd_nn_lanes;
        stats_.planning_time = wall.seconds();
        if (best && stats_.best_cost_events.empty()) stats_.best_cost_events.emplace_back(stats_.first_solution_time,double(best->cost));
        if(best && config_.stop_at_first_solution)
            stats_.closest_goal_distance=double(model_.distanceToOutput(best->state,model_.goalOutput()));
        else for (const auto &node : tree_.nodes())
            stats_.closest_goal_distance =
                std::min(stats_.closest_goal_distance,
                         double(model_.distanceToOutput(node->state, model_.goalOutput())));
        if (!best) {
            if (config_.max_iterations && stats_.iterations >= config_.max_iterations)
                stats_.failure_reason = "max_iterations_no_goal";
            else if (config_.max_nodes && tree_.size() >= config_.max_nodes)
                stats_.failure_reason = "max_nodes_no_goal";
            else
                stats_.failure_reason = stats_.termination_reason == "safety_limit" ? "safety_limit" : "timeout_no_goal";
        }

        if (best) {
            stats_.success = true;
            stats_.cost = best->cost;
            for (Node *n = best; n->parent; n = n->parent)
                path_.push_back(n->incoming);
            std::reverse(path_.begin(), path_.end());

            for (const auto &edge : path_) {
                stats_.duration += edge.T;
                if constexpr(std::is_same_v<Real,double>) {
                auto previous = edge.evaluate(Real(0)).output();
                for (int i = 1; i <= 100; ++i) {
                    const auto p = edge.evaluate(edge.T * Real(i) / Real(100)).output();
                    stats_.path_length += (p - previous).norm();
                    previous = p;
                }
                }
            }
        }
        return stats_;
    }

    const std::vector<Trajectory> &path() const { return path_; }

    const RRTTree<Node> &tree() const { return tree_; }
    const std::vector<Trajectory> &firstPath() const { return first_path_; }

  private:
    struct CachedGoal {
        Edge edge;
        bool rejected{false};
    };
    struct GoalCacheEntry {
        std::vector<CachedGoal> candidates;
        bool connected{false};
        Real connected_parent_cost{std::numeric_limits<Real>::infinity()};
    };
    struct Candidate {
        Node *parent;
        Edge edge;
        Real total;
        std::size_t sequence;
        CachedGoal *cached{nullptr};
    };

    bool goalComplete(const Node *node) const {
        const auto found = goal_cache_.find(node->id);
        if (found == goal_cache_.end())
            return false;
        const auto &entry = found->second;
        // A rewire can lower g(parent). Reopen a successful connection then,
        // because the previous goal node may itself have been reparented.
        return (entry.connected && node->cost >= entry.connected_parent_cost - Real(1e-7)) ||
               std::all_of(entry.candidates.begin(), entry.candidates.end(),
                           [](const auto &c) { return c.rejected; });
    }
    void rejectCached(Candidate &candidate) {
        if (!candidate.cached)
            return;
        candidate.cached->rejected = true;
        if (goalComplete(candidate.parent))
            ++stats_.goal_cache_exhausted;
    }
    void appendCachedGoals(const std::vector<Node *> &parents, std::vector<Candidate> &out,
                           const Timer &wall) {
        if constexpr (HasGoalConstraint<Model>::value) {
            std::array<Node *, BatchWidth> missing{};
            std::array<Constraint, BatchWidth> targets{};
            std::size_t used = 0;
            auto flush = [&] {
                if (!used)
                    return;
                std::vector<Candidate> fresh;
                appendBatch(missing, targets, used, fresh);
                // Empty results are also cached. Node state and metadata are
                // immutable under rewiring; only the current parent cost changes.
                for (std::size_t i = 0; i < used; ++i)
                    goal_cache_.try_emplace(missing[i]->id);
                for (auto &candidate : fresh)
                    goal_cache_.at(candidate.parent->id)
                        .candidates.push_back({std::move(candidate.edge), false});
                used = 0;
            };
            for (Node *parent : parents) {
                if (wall.seconds() >= config_.timeout)
                    break;
                if (goal_cache_.find(parent->id) != goal_cache_.end()) {
                    ++stats_.goal_cache_hits;
                    continue;
                }
                missing[used] = parent;
                targets[used++] = model_.goalConstraint();
                if (used == BatchWidth)
                    flush();
            }
            flush();
            for (Node *parent : parents) {
                auto found = goal_cache_.find(parent->id);
                if (found == goal_cache_.end())
                    continue;
                for (auto &candidate : found->second.candidates)
                    if (!candidate.rejected)
                        out.push_back({parent, candidate.edge, parent->cost + candidate.edge.cost,
                                       out.size(), &candidate});
            }
        }
    }

    void appendBatch(const std::array<Node *, BatchWidth> &parents,
                     const std::array<Constraint, BatchWidth> &targets, std::size_t count,
                     std::vector<Candidate> &out) {
        Timer timer;
        std::array<State, BatchWidth> states{};
        for (std::size_t i = 0; i < count; ++i)
            states[i] = parents[i]->state;
        const auto before = stats_.simd_pff_lanes;
        auto edges = solveBatch(states, targets, count);
        const double elapsed = timer.seconds();
        stats_.pff_time += elapsed;
        if (stats_.simd_pff_lanes > before)
            stats_.simd_time += elapsed;
        stats_.pff_query_count += count;
        for (std::size_t i = 0; i < count; ++i) {
            stats_.pff_eval_count += edges[i].size();
            if (edges[i].empty())
                ++stats_.pff_rejected;
            for (auto &edge : edges[i]) {
                if (!edge.deferred &&
                    !model_.prepareEdge(parents[i]->state, parents[i]->metadata, edge.trajectory)) {
                    ++stats_.rejected_candidates;
                    continue;
                }
                const Real total = parents[i]->cost + edge.cost;
                out.push_back({parents[i], std::move(edge), total, out.size()});
            }
        }
    }

    std::array<std::vector<Edge>, BatchWidth>
    solveBatch(const std::array<State, BatchWidth> &states,
               const std::array<Constraint, BatchWidth> &targets, std::size_t count) {
        auto edges = batch_pff_.candidatesBatch(states, targets, count,
                                                config_.simd_enabled && config_.simd_pff,
                                                &stats_.simd_pff_lanes);
        if constexpr (HasCandidateTimes<Model>::value) {
            if (timing_pff_ && timing_pff_->config().free_terminal_time) {
                for (std::size_t i = 0; i < count; ++i) {
                    for (double time : model_.additionalCandidateTimes(states[i], targets[i])) {
                        if (!std::isfinite(time) || time < timing_pff_->config().min_time ||
                            time > timing_pff_->config().max_time)
                            continue;
                        if (std::any_of(edges[i].begin(), edges[i].end(), [&](const auto &edge) {
                                return std::abs(edge.T - time) <= 1e-9 * (1 + time);
                            }))
                            continue;
                        auto extra =
                            timing_pff_->resultsAtTimes(states[i], targets[i], {time}, false);
                        for (auto &edge : extra)
                            edges[i].push_back(std::move(edge));
                    }
                    // Rewire consumes each parent's results in increasing edge cost.
                    std::stable_sort(edges[i].begin(), edges[i].end(),
                                     [](const auto &a, const auto &b) {
                                         return a.cost == b.cost ? a.T < b.T : a.cost < b.cost;
                                     });
                }
            }
        }
        for (std::size_t i = 0; i < count; ++i)
            for (const auto &edge : edges[i])
                if (!edge.deferred)
                    ++stats_.pff_materializations;
        return edges;
    }

    void sortCandidates(std::vector<Candidate> &c) const {
        std::sort(c.begin(), c.end(), [](const auto &a, const auto &b) {
            return a.total == b.total ? a.sequence < b.sequence : a.total < b.total;
        });
    }

    bool precheckEdge(const Trajectory &tr) {
        if constexpr (HasCandidatePrecheck<Model>::value) {
            ++stats_.candidate_prechecks;
            Timer timer;
            const bool valid = model_.precheckTrajectory(tr, stats_);
            stats_.validation_time += timer.seconds();
            if (!valid)
                ++stats_.candidate_precheck_rejected;
            return valid;
        }
        return true;
    }
    bool materializeEdge(Edge &edge, Node *parent) {
        if (edge.deferred) {
            Timer timer;
            const bool valid = batch_pff_.materialize(parent->state, edge);
            stats_.pff_time += timer.seconds();
            if (!valid) {
                ++stats_.pff_rejected;
                return false;
            }
            ++stats_.pff_materializations;
        }
        return model_.prepareEdge(parent->state, parent->metadata, edge.trajectory);
    }
    bool validateEdge(const Trajectory &tr, bool prechecked = false) {
        if constexpr (HasTrajectoryScreen<Model>::value) {
            if (stats_.simd_active && config_.simd_screen) {
                Timer timer;
                const bool keep = model_.screenTrajectory(tr, config_.collision_samples,
                                                          config_.interleaved_samples, stats_);
                const double elapsed = timer.seconds();
                stats_.screen_time += elapsed;
                stats_.simd_time += elapsed;
                if (!keep) {
                    ++stats_.screen_rejected;
                    return false;
                }
            }
        }
        Timer timer;
        if constexpr (HasCandidatePrecheck<Model>::value) {
            if (prechecked) {
                const bool valid = model_.trajectoryValidAfterPrecheck(tr, stats_);
                stats_.validation_time += timer.seconds();
                return valid;
            }
        }
        const bool valid = model_.trajectoryValid(tr, stats_);
        stats_.validation_time += timer.seconds();
        return valid;
    }

    void rewire(Node *added, const std::vector<Node *> &near, const Timer &wall) {
        std::array<Node *, BatchWidth> nodes{};
        std::array<Constraint, BatchWidth> targets{};
        std::size_t count = 0;
        auto flush = [&] {
            if (!count)
                return;
            std::array<State, BatchWidth> states{};
            for (std::size_t i = 0; i < count; ++i)
                states[i] = added->state;
            Timer timer;
            const auto before = stats_.simd_pff_lanes;
            auto results = solveBatch(states, targets, count);
            const double elapsed = timer.seconds();
            stats_.pff_time += elapsed;
            if (stats_.simd_pff_lanes > before)
                stats_.simd_time += elapsed;
            stats_.pff_query_count += count;
            for (std::size_t lane = 0; lane < count && wall.seconds() < config_.timeout; ++lane) {
                Node *node = nodes[lane];
                stats_.pff_eval_count += results[lane].size();
                for (auto &edge : results[lane]) {
                    if (wall.seconds() >= config_.timeout)
                        break;
                    // Node state and branch never change. Descendant edges are
                    // reused; only cumulative costs propagate after commit.
                    if (added->cost + edge.cost >= node->cost - Real(1e-7))
                        break;
                    if (!materializeEdge(edge, added))
                        continue;
                    const bool precheck =
                        config_.candidate_precheck && HasCandidatePrecheck<Model>::value;
                    if (precheck && !precheckEdge(edge.trajectory))
                        continue;
                    if (validateEdge(edge.trajectory, precheck) &&
                        Rewiring<Node>::commit(node, added, edge.trajectory, model_)) {
                        ++stats_.rewire_count;
                        break;
                    }
                }
            }
            count = 0;
        };
        for (Node *node : near) {
            if (wall.seconds() >= config_.timeout)
                return;
            if (!node->parent || node == added || Rewiring<Node>::isAncestor(node, added))
                continue;
            // Every edge has positive time and nonnegative effort. This is an
            // admissible zero edge-cost lower bound, applied BEFORE solving.
            if (added->cost >= node->cost - Real(1e-7))
                continue;
            nodes[count] = node;
            targets[count] = Constraint::full(node->state);
            if (++count == BatchWidth)
                flush();
        }
        flush();
    }

    Node *bestGoal() const {
        Node *best = nullptr;
        for (Node *n : goals_)
            if (!best || n->cost < best->cost)
                best = n;
        return best;
    }

    void pruneTree() {
        Node *best = bestGoal();
        if (!best)
            return;

        std::vector<Node *> removed;
        for (const auto &owned : tree_.nodes()) {
            Node *n = owned.get();
            if (n == tree_.root() || !n->children.empty() || Rewiring<Node>::isAncestor(n, best))
                continue;
            if (std::find(goals_.begin(), goals_.end(), n) != goals_.end())
                continue;
            // An admissible model bound from START via n to the goal is needed;
            // current tree g(n) is not a valid global lower bound for pruning.
            if (model_.solutionLowerBound(n->state) >= best->cost)
                removed.push_back(n);
        }
        for (Node *n : removed)
            tree_.removeNode(n);
    }

    Model model_;
    PlannerConfigT<Real> config_;
    std::optional<Solver> timing_pff_;
    BatchSolver batch_pff_;
    RRTTree<Node> tree_;
    std::vector<Node *> goals_;
    std::vector<Trajectory> path_, first_path_;
    PlanResult stats_{};
    std::unordered_map<std::size_t, GoalCacheEntry> goal_cache_;
};
} // namespace rrtpff
