from v2_analysis import analyse


def _row(seed: int, planner: str, success: bool) -> dict[str, str]:
    return {
        "condition": "neutral",
        "seed": str(seed),
        "planner": planner,
        "mission_success": str(success),
        "path_length": "10",
        "route_risk": "1",
        "min_recoverability": "0.5",
    }


def test_analysis_preserves_pairing_and_discordance() -> None:
    rows = []
    for seed in range(2):
        rows.extend(
            [
                _row(seed, "J0_shortest", seed == 1),
                _row(seed, "J1_risk", True),
                _row(seed, "J2_recoverability", True),
                _row(seed, "J3_joint", True),
            ]
        )
    summaries, contrasts = analyse(rows)
    assert len(summaries) == 4
    primary = next(row for row in contrasts if row["baseline"] == "J0_shortest")
    assert primary["pairs"] == 2
    assert primary["baseline_only_failures"] == 1
    assert primary["candidate_only_failures"] == 0
    assert primary["candidate_minus_baseline_failure_risk"] == -0.5
