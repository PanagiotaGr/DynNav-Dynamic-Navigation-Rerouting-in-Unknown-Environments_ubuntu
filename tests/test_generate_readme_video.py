from __future__ import annotations

from scripts.generate_readme_video import EVENT_CELL, EVENT_STEP, build_demo_run, render_frame


def test_readme_animation_uses_a_successful_replanning_trace() -> None:
    run = build_demo_run()

    assert run.result.success
    assert EVENT_CELL in run.result.steps[0].selected_path
    assert run.result.steps[EVENT_STEP].applied_updates
    assert EVENT_CELL not in run.result.steps[EVENT_STEP].selected_path
    assert EVENT_CELL not in run.result.trajectory
    assert run.result.trajectory[-1] == run.goal


def test_readme_animation_frame_renders_planner_telemetry() -> None:
    run = build_demo_run()
    position = run.result.trajectory[EVENT_STEP]

    frame = render_frame(
        run,
        step=EVENT_STEP,
        robot=position,
        phase="ROUTE INVALIDATED → REPLANNING",
        pulse=1.0,
    )

    assert frame.size == (960, 540)
    assert frame.mode == "RGB"
