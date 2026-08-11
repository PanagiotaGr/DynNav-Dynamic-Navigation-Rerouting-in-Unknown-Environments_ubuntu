"""ROS-independent planner configuration for the static Nav2 benchmark."""

from __future__ import annotations

import copy
from typing import Any

PLANNER_IDS = (
    "NavFn",
    "Smac2D",
    "DynNavShortest",
    "DynNavRisk",
    "DynNavRecoverability",
    "DynNavJoint",
)


def planner_parameter_overrides() -> dict[str, dict[str, Any]]:
    """Return the complete six-planner configuration used in every trial."""

    return {
        "NavFn": {
            "plugin": "nav2_navfn_planner::NavfnPlanner",
            "tolerance": 0.5,
            "use_astar": False,
            "allow_unknown": True,
        },
        "Smac2D": {
            "plugin": "nav2_smac_planner::SmacPlanner2D",
            "tolerance": 0.125,
            "downsample_costmap": False,
            "downsampling_factor": 1,
            "allow_unknown": True,
            "max_iterations": 1_000_000,
            "max_on_approach_iterations": 1_000,
            "terminal_checking_interval": 5_000,
            "max_planning_time": 2.0,
            "cost_travel_multiplier": 2.0,
        },
        "DynNavShortest": {
            "plugin": "dynnav_nav2_cpp::DynNavGlobalPlanner",
            "allow_unknown": True,
            "lethal_cost_threshold": 253,
            "neutral_cost": 1.0,
            "risk_weight": 0.0,
            "irreversibility_weight": 0.0,
            "unknown_risk": 0.5,
            "max_iterations": 0,
        },
        "DynNavRisk": {
            "plugin": "dynnav_nav2_cpp::DynNavGlobalPlanner",
            "allow_unknown": True,
            "lethal_cost_threshold": 253,
            "neutral_cost": 1.0,
            "risk_weight": 4.0,
            "irreversibility_weight": 0.0,
            "unknown_risk": 0.5,
            "max_iterations": 0,
        },
        "DynNavRecoverability": {
            "plugin": "dynnav_nav2_cpp::DynNavGlobalPlanner",
            "allow_unknown": True,
            "lethal_cost_threshold": 253,
            "neutral_cost": 1.0,
            "risk_weight": 0.0,
            "irreversibility_weight": 4.0,
            "unknown_risk": 0.5,
            "max_iterations": 0,
        },
        "DynNavJoint": {
            "plugin": "dynnav_nav2_cpp::DynNavGlobalPlanner",
            "allow_unknown": True,
            "lethal_cost_threshold": 253,
            "neutral_cost": 1.0,
            "risk_weight": 4.0,
            "irreversibility_weight": 4.0,
            "unknown_risk": 0.5,
            "max_iterations": 0,
        },
    }


def inject_planner_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy base Nav2 parameters and replace only planner-server plugins."""

    merged = copy.deepcopy(payload)
    try:
        planner_parameters = merged["planner_server"]["ros__parameters"]
    except (KeyError, TypeError) as exc:
        raise ValueError("base Nav2 parameters do not define planner_server") from exc
    if not isinstance(planner_parameters, dict):
        raise ValueError("planner_server.ros__parameters must be a mapping")

    planner_parameters["planner_plugins"] = list(PLANNER_IDS)
    planner_parameters.pop("GridBased", None)
    planner_parameters.update(planner_parameter_overrides())
    return merged
