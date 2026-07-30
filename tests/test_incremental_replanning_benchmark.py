from dynnav.experiments.incremental_replanning_benchmark import compare_replanning_algorithms
from dynnav.planners.grid_map import GridMap
from dynnav.planners.online_recoverability import ObstacleUpdate


def test_benchmark_compares_same_steps_and_algorithms():
    grid = GridMap.from_obstacles(9, 5)
    rows = compare_replanning_algorithms(
        grid,
        (0, 2),
        (8, 2),
        {1: [ObstacleUpdate((4, 2), True)]},
    )
    assert rows
    assert {row.algorithm for row in rows} == {"full_astar", "incremental_dstar_lite"}
    by_step = {}
    for row in rows:
        by_step.setdefault(row.step, []).append(row)
    assert all(len(step_rows) == 2 for step_rows in by_step.values())


def test_algorithms_agree_on_success_after_update():
    grid = GridMap.from_obstacles(10, 6)
    rows = compare_replanning_algorithms(
        grid,
        (0, 3),
        (9, 3),
        {1: [ObstacleUpdate((4, 3), True)], 2: [ObstacleUpdate((4, 2), True)]},
    )
    by_step = {}
    for row in rows:
        by_step.setdefault(row.step, {})[row.algorithm] = row
    for algorithms in by_step.values():
        assert algorithms["full_astar"].success == algorithms["incremental_dstar_lite"].success
