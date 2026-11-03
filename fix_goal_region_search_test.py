#!/usr/bin/env python3
"""Update the legacy goal-connector test to match Goal-region-triggered PFF.

This changes tests/test_search_improvements.cpp only. It does not alter planner,
PFF, SIMD, rewiring, YAML configurations, or Shortcut.
Usage: python3 fix_goal_region_search_test.py [project_directory]
"""
from pathlib import Path
from datetime import datetime
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').expanduser().resolve()
p = root / 'tests/test_search_improvements.cpp'
if not p.is_file():
    raise SystemExit(f'Missing source file: {p}')
s = p.read_text(encoding='utf-8')
old_comment = '''        // A remote full-state goal is attempted without changing ordinary range
        // limiting. The same curve must still be rejected if an obstacle blocks it.'''
new_comment = '''        // An ordinary extension still obeys range. A node entering the Goal region
        // triggers a separate validated PFF connection to the full Goal state.'''
old = '''        pc.goal_tolerance = 1e-7;
        for (bool simd : {false, true}) {
            pc.simd_enabled = simd;
            RRTstar<GoalModel> planner(GoalModel(mc), pc);
            auto result = planner.solve();
            require(result.success && result.node_count == 2 && result.goal_connection_queries == 1,
                    "explicit goal connector bypasses range and fixes terminal state");
            require(planner.path().back().end.derivative(1).norm() < 1e-8, "goal velocity fixed");
            pc.explicit_goal_connection = false;
            RRTstar<GoalModel> limited(GoalModel(mc), pc);
            auto blocked = limited.solve();
            require(!blocked.success && blocked.goal_connection_queries == 0,
                    "ordinary expansion retains range limit");
            require(blocked.failure_reason == "max_iterations_no_goal", "termination reason");
            pc.explicit_goal_connection = true;
        }
        auto blocked_config = mc;'''
new = '''        pc.goal_tolerance = 1e-7;
        for (bool simd : {false, true}) {
            pc.simd_enabled = simd;

            // One ordinary range-limited extension ends about 1.9 units from Goal.
            pc.goal_connect_radius = 0.5;
            RRTstar<GoalModel> outside(GoalModel(mc), pc);
            auto outside_result = outside.solve();
            require(!outside_result.success && outside_result.node_count == 2 &&
                        outside_result.goal_connection_queries == 0,
                    "no exact Goal connection when new node is outside region");

            // A new node enters the larger region; use a separate full-state PFF
            // edge. Neither sampling nor regular expansion bypasses range.
            pc.goal_connect_radius = 2.0;
            RRTstar<GoalModel> planner(GoalModel(mc), pc);
            auto result = planner.solve();
            require(result.success && result.node_count == 3 &&
                        result.goal_connection_queries == 1 &&
                        result.goal_connection_valid == 1,
                    "region entry triggers one validated exact PFF Goal connection");
            require(planner.path().size() == 2,
                    "normal extension then separate exact Goal edge");
            require((planner.path().front().end.output() - mc.start.output()).norm() <
                        pc.range + 1e-7,
                    "ordinary extension respects range even with Goal bias");
            require(planner.path().back().end.derivative(1).norm() < 1e-8,
                    "exact Goal connection fixes terminal velocity");

            pc.explicit_goal_connection = false;
            RRTstar<GoalModel> limited(GoalModel(mc), pc);
            auto blocked = limited.solve();
            require(!blocked.success && blocked.goal_connection_queries == 0,
                    "disabling explicit Goal connection retains range limit");
            require(blocked.failure_reason == "max_iterations_no_goal", "termination reason");
            pc.explicit_goal_connection = true;
        }
        auto blocked_config = mc;'''
for before, after, label in ((old_comment, new_comment, 'comment'), (old, new, 'Goal test block')):
    count=s.count(before)
    if count != 1:
        raise SystemExit(f'Expected one {label} anchor, found {count}; no changes made')
    s=s.replace(before,after,1)
backup=p.with_name(p.name + '.backup_goal_test_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.write_text(p.read_text(encoding='utf-8'),encoding='utf-8')
p.write_text(s,encoding='utf-8')
print('Updated:',p)
print('Backup:',backup)
print('Now run: cmake --build build --target test_search_improvements -j4')
print('Then: ctest --test-dir build -R "^search_improvements$" --output-on-failure')
