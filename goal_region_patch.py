#!/usr/bin/env python3
"""Install per-node Goal region PFF connection; keep the RRT* / Shortcut pipeline.

Usage: python3 goal_region_patch.py /absolute/path/to/project

This script edits only local files. It creates timestamped backups, and aborts
without changing files if code anchors are not found exactly once.
"""
from pathlib import Path
import sys
import datetime
import re

CASES = [
    'uni_bugtrap', 'uni_wall', 'quad2d_bugtrap', 'quad2d_hole',
    'quad3d_block', 'quad3d_window'
]

def once(s, old, new, name):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'{name}: expected one anchor, found {n}; no files changed')
    return s.replace(old, new, 1)

def update_config(s):
    s = once(s, '    Real goal_tolerance{Real(.05)};\n',
             '    Real goal_tolerance{Real(.05)};\n'
             '    Real goal_connect_radius{Real(1)}; // Flat-output distance; not success tolerance\n',
             'config radius')
    s = once(s, '    std::size_t goal_parent_limit{16};\n', '', 'old goal limit field')
    s = once(s, '|| !goal_parent_limit ||',
             '|| !std::isfinite(goal_connect_radius) || goal_connect_radius <= Real(0) ||',
             'validate goal radius')
    return s

def update_runner(s):
    return once(s,
        '    c.goal_parent_limit=n["goal_parent_limit"].as<std::size_t>(16);',
        '    c.goal_connect_radius=n["goal_connect_radius"].as<float>(1.0f);',
        'parse goal radius')

def update_rrt(s):
    s = once(s,
        '            bool sampled_goal = false;\n'
        '            const Output target = sampler.sampleFlatOutput(config_.goal_bias, &sampled_goal);\n'
        '            const bool goal_connection =\n'
        '                sampled_goal && config_.explicit_goal_connection && HasGoalConstraint<Model>::value;',
        '            // Goal bias is retained. Exact goal edges are tried from the region.\n'
        '            const Output target = sampler.sampleFlatOutput(config_.goal_bias);',
        'replace goal-special sampling')
    s = once(s, '            const Output limited =\n'
        '                goal_connection ? target\n'
        '                                : model_.limitExtension(nearest->state, target, config_.range);',
        '            const Output limited =\n'
        '                model_.limitExtension(nearest->state, target, config_.range);',
        'remove special goal bypass of extension range')
    s = once(s,
        '            if (!goal_connection && parents.size() < config_.minimum_parent_candidates) {',
        '            if (parents.size() < config_.minimum_parent_candidates) {',
        'restore minimum parents for goal-biased sampling')
    start = '            if (goal_connection) {\n                // Bounded goal-parent set:'
    end = '\n            stats_.nn_time += nn_timer.seconds();'
    if s.count(start) != 1 or s.count(end) != 1:
        raise RuntimeError('goal Top-K block could not be isolated; no files changed')
    i, j = s.index(start), s.index(end, s.index(start))
    s = s[:i] + s[j:]
    s = once(s,
        '            if (goal_connection && config_.cache_goal_connections)\n'
        '                appendCachedGoals(parents, candidates, wall);\n'
        '            else\n'
        '                for (std::size_t offset = 0; offset < parents.size(); offset += BatchWidth) {',
        '            for (std::size_t offset = 0; offset < parents.size(); offset += BatchWidth) {',
        'remove special Goal candidate cache branch')
    s = once(s,
        '                        if constexpr (HasGoalConstraint<Model>::value)\n'
        '                            if (goal_connection)\n'
        '                                targets[i] = model_.goalConstraint();\n',
        '', 'remove special Goal Constraint branch')
    # Both alternative-constraint branches now run normally for all targets.
    old = '                        if (!goal_connection) {'
    if s.count(old) != 2:
        raise RuntimeError('expected exactly 2 ordinary-expansion guards; no files changed')
    s = s.replace(old, '                        {')
    s = once(s,
        '''                if (!materializeEdge(candidate.edge, candidate.parent)) {
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
                    ++stats_.goal_connection_valid;''',
        '''                if (!materializeEdge(candidate.edge, candidate.parent))
                    continue;
                if (precheck && !precheckEdge(candidate.edge.trajectory))
                    continue;
                if (!validateEdge(candidate.edge.trajectory, precheck))
                    continue;''',
        'remove cached goal logic from ordinary edge materialization')
    old_goal = '''            if (model_.isGoal(added->state, config_.goal_tolerance)) {
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
            }'''
    new_goal = '''            // A radius hit is only a trigger. Success requires a validated
            // PFF trajectory ending at the full flat Goal boundary.
            Node *goal_node = model_.isGoal(added->state, config_.goal_tolerance)
                                  ? added : connectGoalFromRegion(added, wall);
            if (goal_node) {
                goals_.push_back(goal_node);
                if (!std::isfinite(stats_.first_solution_time)) {
                    stats_.first_solution_time = wall.seconds();
                    stats_.first_solution_cost = double(goal_node->cost);
                    stats_.best_cost_events.emplace_back(stats_.first_solution_time,double(goal_node->cost));
                    for (Node *n=goal_node;n->parent;n=n->parent) first_path_.push_back(n->incoming);
                    std::reverse(first_path_.begin(),first_path_.end());
                }
                if (config_.stop_at_first_solution) {
                    stats_.termination_reason = "first_solution";
                    break;
                }
            }'''
    s = once(s, old_goal, new_goal, 'goal handling and first solution snapshot')
    helper = '''    // One exact PFF connection per newly added node inside the Goal region.
    // This is NOT accepting an approximate Goal or making a straight-line jump.
    Node *connectGoalFromRegion(Node *parent, const Timer &wall) {
        if constexpr (!HasGoalConstraint<Model>::value) {
            return nullptr;
        } else {
            if (!config_.explicit_goal_connection ||
                wall.seconds() >= config_.timeout ||
                model_.distanceToOutput(parent->state, model_.goalOutput()) >
                    config_.goal_connect_radius ||
                (config_.max_nodes && tree_.size() >= config_.max_nodes) ||
                (config_.hard_safety_node_cap &&
                 tree_.size() >= config_.hard_safety_node_cap))
                return nullptr;

            ++stats_.goal_connection_queries;
            std::vector<Node *> only_parent{parent};
            std::vector<Candidate> goal_candidates;
            if (config_.cache_goal_connections) {
                appendCachedGoals(only_parent, goal_candidates, wall);
            } else {
                std::array<Node *, BatchWidth> block{};
                std::array<Constraint, BatchWidth> targets{};
                block[0] = parent;
                targets[0] = model_.goalConstraint();
                appendBatch(block, targets, 1, goal_candidates);
            }
            sortCandidates(goal_candidates);
            const bool precheck = config_.candidate_precheck &&
                                  HasCandidatePrecheck<Model>::value;
            for (auto &candidate : goal_candidates) {
                if (wall.seconds() >= config_.timeout)
                    break;
                if (!materializeEdge(candidate.edge, parent)) {
                    rejectCached(candidate);
                    continue;
                }
                if (candidate.cached)
                    candidate.cached->edge = candidate.edge;
                if (!model_.isGoal(candidate.edge.terminal_state,
                                   config_.goal_tolerance)) {
                    rejectCached(candidate);
                    continue;
                }
                if (precheck && !precheckEdge(candidate.edge.trajectory)) {
                    rejectCached(candidate);
                    continue;
                }
                if (!validateEdge(candidate.edge.trajectory, precheck)) {
                    rejectCached(candidate);
                    continue;
                }
                if (candidate.cached) {
                    auto &entry = goal_cache_.at(parent->id);
                    entry.connected = true;
                    entry.connected_parent_cost = parent->cost;
                }
                auto *goal = tree_.createNode(candidate.edge.terminal_state,
                                              parent, candidate.edge.trajectory);
                goal->metadata = model_.terminalMetadata(candidate.edge.trajectory);
                ++stats_.goal_connection_valid;
                return goal;
            }
            return nullptr;
        }
    }

'''
    s = once(s, '    void sortCandidates(std::vector<Candidate> &c) const {',
             helper + '    void sortCandidates(std::vector<Candidate> &c) const {',
             'insert region PFF connector')
    if 'goal_parent_limit' in s or 'std::min(config_.goal_parent_limit' in s:
        raise RuntimeError('Old Goal Top-K code still exists; no files changed')
    return s

def update_case(s, case):
    s = once(s, '  goal_parent_limit: 16',
             '  goal_connect_radius: 1.0', case + ' radius YAML')
    # If planner success tolerance is looser than the final path validator,
    # an approximate Goal can be declared solved and later fail validation.
    a = re.search(r'(?m)^  goal_tolerance:\s*([0-9.eE+-]+)\s*$', s)
    b = re.search(r'(?m)^  endpoint_tolerance:\s*([0-9.eE+-]+)\s*$', s)
    if not a or not b:
        raise RuntimeError(case + ': cannot verify endpoint tolerance')
    if float(a.group(1)) > float(b.group(1)):
        s = once(s, a.group(0), '  goal_tolerance: 0.000001',
                 case + ': ensure Goal success tolerance is strict')
    return s

def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 goal_region_patch.py /absolute/path/to/project')
    root = Path(sys.argv[1]).expanduser().resolve()
    sources = {
        root / 'common/planner_config.h': update_config,
        root / 'benchmark/production_runner.cpp': update_runner,
        root / 'core/rrtstar.h': update_rrt,
    }
    for case in CASES:
        sources[root / 'benchmark/config' / (case + '.yaml')] = \
            lambda s, case=case: update_case(s, case)
    if any(not path.is_file() for path in sources):
        missing = [str(p) for p in sources if not p.is_file()]
        raise SystemExit('Missing files; no files changed: ' + ', '.join(missing))
    changes = []
    for path, func in sources.items():
        old = path.read_text(encoding='utf-8')
        if 'connectGoalFromRegion' in old and path.name == 'rrtstar.h':
            raise SystemExit('Goal region already appears patched; no files changed')
        new = func(old)
        changes.append((path, old, new))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for path, old, new in changes:
        backup = path.with_name(path.name + '.backup_goal_region_' + stamp)
        backup.write_text(old, encoding='utf-8')
        path.write_text(new, encoding='utf-8')
        print('Updated', path.relative_to(root), '(backup:', backup.name + ')')
    print('Done. This is a new algorithm variant; rebuild benchmark and run tests.')
    print('IMPORTANT: keep goal_tolerance <= validation.endpoint_tolerance per case.')

if __name__ == '__main__':
    main()
