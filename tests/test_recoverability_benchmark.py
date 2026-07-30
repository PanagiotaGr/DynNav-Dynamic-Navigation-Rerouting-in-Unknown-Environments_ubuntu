from dynnav.experiments.recoverability_benchmark import run_benchmark
from dynnav.planners.recoverability_astar import PlannerMode


def test_benchmark_emits_all_modes_for_each_seed(tmp_path):
    output = tmp_path / "recoverability.csv"
    records = run_benchmark(range(3), output)

    assert len(records) == 3 * len(PlannerMode)
    assert output.exists()
    assert {record.mode for record in records} == {mode.value for mode in PlannerMode}


def test_benchmark_metrics_are_normalized_and_auditable():
    records = run_benchmark([0])

    for record in records:
        assert record.geometric_length >= 0
        assert record.path_length_overhead >= 0.0
        assert record.cumulative_risk >= 0.0
        assert record.cumulative_irreversibility >= 0.0
        assert record.minimum_escape_options >= 0
        assert record.nodes_expanded >= 0
        assert record.planning_time_ms >= 0.0
