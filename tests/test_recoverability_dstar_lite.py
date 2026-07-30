from dynnav.planners.grid_map import GridMap
from dynnav.planners.recoverability_astar import PlannerMode, RecoverabilityAStarConfig, recoverability_astar
from dynnav.planners.recoverability_dstar_lite import RecoverabilityDStarLite


def test_initial_plan_matches_weighted_astar_cost():
    risk = {(3, y): 0.8 for y in range(1, 5)}
    grid = GridMap.from_obstacles(8, 6, risk=risk)
    config = RecoverabilityAStarConfig(risk_weight=3.0, irreversibility_weight=2.0)
    expected = recoverability_astar(
        grid, (0, 2), (7, 2), safe_cells={(0, 2)}, mode=PlannerMode.PROPOSED, config=config
    )
    actual = RecoverabilityDStarLite(
        grid, (0, 2), (7, 2), safe_cells={(0, 2)}, mode=PlannerMode.PROPOSED, config=config
    ).plan()
    assert actual.success
    assert actual.cost == expected.cost


def test_incremental_repair_avoids_new_obstacle():
    grid = GridMap.from_obstacles(10, 6)
    planner = RecoverabilityDStarLite(grid, (0, 2), (9, 2))
    first = planner.plan()
    assert first.success
    blocked = first.path[3]
    planner.update_obstacles({blocked: True})
    repaired = planner.replan()
    assert repaired.success
    assert blocked not in repaired.path
    assert planner.replan_count == 1


def test_moving_start_reuses_value_function():
    grid = GridMap.from_obstacles(12, 8)
    planner = RecoverabilityDStarLite(grid, (0, 0), (11, 7))
    first = planner.plan()
    assert first.success
    next_start = first.path[1]
    repaired = planner.replan(new_start=next_start)
    assert repaired.success
    assert repaired.path[0] == next_start
    assert repaired.path[-1] == (11, 7)


def test_recoverability_mode_can_avoid_fragile_corridor():
    obstacles = {
        (x, y)
        for x in range(1, 8)
        for y in range(1, 6)
        if y not in {2, 5}
    }
    grid = GridMap.from_obstacles(9, 7, obstacles=obstacles)
    shortest = RecoverabilityDStarLite(grid, (0, 2), (8, 2), mode=PlannerMode.SHORTEST).plan()
    proposed = RecoverabilityDStarLite(
        grid,
        (0, 2),
        (8, 2),
        mode=PlannerMode.PROPOSED,
        config=RecoverabilityAStarConfig(irreversibility_weight=10.0),
    ).plan()
    assert shortest.success and proposed.success
    assert proposed.cumulative_irreversibility <= shortest.cumulative_irreversibility


def test_invalid_update_is_rejected():
    planner = RecoverabilityDStarLite(GridMap.from_obstacles(4, 4), (0, 0), (3, 3))
    try:
        planner.update_obstacles({(9, 9): True})
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-bounds update should fail")
