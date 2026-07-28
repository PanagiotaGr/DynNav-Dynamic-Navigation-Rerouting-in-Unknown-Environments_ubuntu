from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml

from dynnav_dashboard.config import ScenarioConfig
from dynnav_dashboard.simulation import build_environment, plan_astar, plan_risk_aware, simulate_rollout

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "presentation.yaml"


def load_presentation_config(path: Path = CONFIG_PATH) -> ScenarioConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(ScenarioConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown ScenarioConfig keys: {', '.join(unknown)}")
    for key in ("start", "goal"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return ScenarioConfig(**raw)


st.set_page_config(page_title="DynNav Presentation Mode", page_icon="🧭", layout="wide")
st.title("DynNav — Guided Presentation Mode")
st.caption("A deterministic, presentation-focused walkthrough of risk-aware dynamic navigation.")
st.info(
    "Synthetic research demonstration. Results support controlled software evaluation only; "
    "they do not establish certified safety or physical-robot validation."
)

try:
    cfg = load_presentation_config()
except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
    st.error(f"Could not load {CONFIG_PATH.relative_to(ROOT)}: {exc}")
    st.stop()

env = build_environment(cfg, seed=cfg.random_seed)
baseline = plan_astar(env, cfg.start, cfg.goal)
risk_plan = plan_risk_aware(env, cfg.start, cfg.goal, cfg.risk_weight)
rollout = simulate_rollout(env, cfg, use_risk_aware=True, dynamic_step_every=2)

if not rollout.frames:
    st.error("The canonical scenario produced no simulation frames.")
    st.stop()

with st.sidebar:
    st.header("Presentation controls")
    stage = st.radio(
        "Narrative stage",
        [
            "1 · Research problem",
            "2 · Baseline comparison",
            "3 · Dynamic response",
            "4 · Evidence and limitations",
        ],
    )
    frame_index = st.slider("Simulation step", 0, len(rollout.frames) - 1, 0)
    st.caption(f"Canonical seed: {cfg.random_seed}")
    st.caption("Scenario: configs/presentation.yaml")

frame = rollout.frames[frame_index]
robot_x, robot_y = frame.robot
local_risk = float(frame.risk_snapshot[robot_y, robot_x])
local_uncertainty = float(env.uncertainty[robot_y, robot_x])

if rollout.reached_goal and frame.step == rollout.frames[-1].step:
    supervisor = "GOAL REACHED"
elif not frame.path_remaining:
    supervisor = "SAFE STOP"
elif frame.replanned:
    supervisor = "REPLAN"
elif local_risk >= 0.70 or local_uncertainty >= 0.75:
    supervisor = "CAUTION"
else:
    supervisor = "NORMAL"

if stage.startswith("1"):
    st.header("Research problem")
    st.markdown(
        "A shortest-path planner can select a geometrically efficient route that is fragile under "
        "uncertainty or dynamic change. DynNav evaluates routes using geometric cost, risk exposure, "
        "online route validity, and supervisor actions."
    )
    st.code(
        "observe → estimate uncertainty/risk → plan → monitor → continue/replan/stop → record evidence",
        language="text",
    )
elif stage.startswith("2"):
    st.header("Baseline comparison")
    st.markdown(
        "The comparison isolates the trade-off between geometric efficiency and risk-aware route selection "
        "under the same deterministic environment."
    )
    st.dataframe(
        {
            "metric": ["success", "path cells", "expanded nodes", "runtime ms", "cost", "average risk", "maximum risk"],
            "Classical A*": [
                baseline.success,
                len(baseline.path),
                baseline.expansions,
                round(baseline.runtime_ms, 3),
                round(baseline.cost, 3),
                round(baseline.avg_risk, 3),
                round(baseline.max_risk, 3),
            ],
            "Risk-aware A*": [
                risk_plan.success,
                len(risk_plan.path),
                risk_plan.expansions,
                round(risk_plan.runtime_ms, 3),
                round(risk_plan.cost, 3),
                round(risk_plan.avg_risk, 3),
                round(risk_plan.max_risk, 3),
            ],
        },
        hide_index=True,
        use_container_width=True,
    )
elif stage.startswith("3"):
    st.header("Dynamic response")
    st.markdown(
        "Move through the rollout to show route monitoring, obstacle motion, replanning, and the resulting supervisor state."
    )
else:
    st.header("Evidence and limitations")
    st.markdown(
        "**Supported here:** deterministic software execution, planner comparison, route monitoring, dynamic replanning, "
        "risk metrics, and event-level inspection.\n\n"
        "**Not established here:** formal end-to-end safety, ROS 2 deployment, Gazebo validation, hardware reliability, "
        "or generalisation to arbitrary real environments."
    )

metrics = st.columns(6)
metrics[0].metric("Step", frame.step)
metrics[1].metric("Supervisor", supervisor)
metrics[2].metric("Replans", frame.replan_count)
metrics[3].metric("Local risk", f"{local_risk:.3f}")
metrics[4].metric("Uncertainty", f"{local_uncertainty:.3f}")
metrics[5].metric("Remaining cells", len(frame.path_remaining))

plot_col, interpretation_col = st.columns([2.2, 1])
with plot_col:
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    occupancy = np.clip(env.static + frame.dynamic_snapshot, 0, 1)
    ax.imshow(occupancy, origin="lower", cmap="Greys", alpha=0.90)
    ax.imshow(env.uncertainty, origin="lower", cmap="Purples", alpha=0.22)
    ax.imshow(frame.risk_snapshot, origin="lower", cmap="Oranges", alpha=0.28)
    if baseline.path:
        bx, by = zip(*baseline.path)
        ax.plot(bx, by, "--", linewidth=1.5, label="Initial A* route")
    if frame.path_remaining:
        px, py = zip(*frame.path_remaining)
        ax.plot(px, py, linewidth=2.8, label="Active DynNav route")
    dyn_y, dyn_x = np.where(frame.dynamic_snapshot > 0.5)
    if len(dyn_x):
        ax.scatter(dyn_x, dyn_y, marker="s", s=38, label="Moving obstacle")
    ax.scatter(robot_x, robot_y, marker="o", s=130, label="Robot")
    ax.scatter(*cfg.goal, marker="*", s=180, label="Goal")
    ax.set_title(f"Canonical presentation scenario — step {frame.step}")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    st.pyplot(fig, clear_figure=True)

with interpretation_col:
    explanations = {
        "NORMAL": "The monitored route remains traversable and measured signals remain below caution thresholds.",
        "CAUTION": "The route remains available, but local risk or uncertainty is elevated.",
        "REPLAN": "Dynamic change compromised the active route and triggered a new plan.",
        "SAFE STOP": "No usable route is available, so motion is withheld.",
        "GOAL REACHED": "The robot reached the goal and the episode terminated.",
    }
    st.subheader(supervisor)
    st.write(explanations[supervisor])
    st.progress(min(max(local_risk, 0.0), 1.0), text=f"Risk: {local_risk:.3f}")
    st.progress(min(max(local_uncertainty, 0.0), 1.0), text=f"Uncertainty: {local_uncertainty:.3f}")
    st.json(
        {
            "reached_goal": rollout.reached_goal,
            "total_distance": round(rollout.total_distance, 3),
            "total_replans": rollout.total_replans,
            "average_risk": round(rollout.avg_risk, 3),
            "maximum_risk": round(rollout.max_risk, 3),
            "average_compute_ms": round(rollout.avg_compute_ms, 3),
        }
    )

st.divider()
st.code("streamlit run app/presentation.py", language="bash")
