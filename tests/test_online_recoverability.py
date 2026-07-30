from dynnav.planners.grid_map import GridMap
from dynnav.planners.online_recoverability import (
    EpisodeStatus,
    ObstacleUpdate,
    OnlineRecoverabilityPlanner,
)
from dynnav.planners.recoverability_astar import PlannerMode, RecoverabilityAStarConfig


def test_online_replanner_reaches_goal_without_updates():
    grid = GridMap.from_obstacles(6, 4)
    result = OnlineRecoverabilityPlanner(grid, (0, 1), (5, 1)).run()

    assert result.success
    assert result.status is EpisodeStatus.SUCCESS
    assert result.trajectory[0] == (0, 1)
    assert result.trajectory[-1] == (5, 1)
    assert result.path_length == 5
    assert result.total_nodes_expanded > 0


def test_route_invalidation_triggers_replan_and_avoids_blocked_cell():
    grid = GridMap.from_obstacles(7, 5)
    schedule = {2: [ObstacleUpdate((3, 2), blocked=True)]}
    result = OnlineRecoverabilityPlanner(grid, (0, 2), (6, 2)).run(schedule)

    assert result.success
    assert (3, 2) not in result.trajectory
    assert result.replanning_count >= 1
    assert any(step.applied_updates for step in result.steps)


def test_irreversible_failure_is_distinguished_from_recoverable_invalidation():
    obstacles = {(1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2)}
    grid = GridMap.from_obstacles(5, 3, obstacles=obstacles)
    schedule = {2: [ObstacleUpdate((1, 1), blocked=True), ObstacleUpdate((3, 1), blocked=True)]}
    result = OnlineRecoverabilityPlanner(grid, (0, 1), (4, 1)).run(schedule)

    assert not result.success
    assert result.status is EpisodeStatus.IRREVERSIBLE_FAILURE
    assert result.irreversible_failure


def test_proposed_mode_can_prefer_open_detour_over_fragile_corridor():
    obstacles = {
        (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 2), (2, 2), (3, 2), (4, 2),
    }
    grid = GridMap.from_obstacles(7, 5, obstacles=obstacles)
    config = RecoverabilityAStarConfig(risk_weight=0.0, irreversibility_weight=8.0)

    shortest = OnlineRecoverabilityPlanner(
        grid, (0, 1), (6, 1), mode=PlannerMode.SHORTEST, config=config
    ).run()
    proposed = OnlineRecoverabilityPlanner(
        grid, (0, 1), (6, 1), mode=PlannerMode.PROPOSED, config=config
    ).run()

    assert shortest.success and proposed.success
    assert shortest.minimum_escape_options <= proposed.minimum_escape_options


def test_invalid_update_is_rejected():
    grid = GridMap.from_obstacles(4, 4)
    planner = OnlineRecoverabilityPlanner(grid, (0, 0), (3, 3))

    try:
        planner.run({0: [ObstacleUpdate((9, 9))]})
    except ValueError as exc:
        assert "outside grid" in str(exc)
    else:
        raise AssertionError("expected invalid occupancy update to fail")
