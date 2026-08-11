"""Canonical DynNav research package.

Algorithms remain independent of ROS 2; middleware adapters live in the ROS
workspace and consume these typed, deterministically tested APIs.
"""

from dynnav.config import DynNavConfig
from dynnav.core import (
    GridMap,
    NavigationState,
    PathEvaluation,
    Pose,
    SelfAwareCostWeights,
    SelfAwarenessScore,
    Trajectory,
    estimate_self_awareness,
    expected_information_gain,
    self_aware_path_cost,
)
from dynnav.research_modules import (
    DynNavResearchStack,
    MissionRiskEstimator,
    MissionRiskReport,
    RuntimeMonitor,
    RuntimeObservation,
    SafeModeSupervisor,
    SafetyMode,
    UncertaintyPropagator,
    UncertaintyState,
)

__version__ = "0.2.0"

__all__ = [
    "DynNavConfig",
    "DynNavResearchStack",
    "GridMap",
    "MissionRiskEstimator",
    "MissionRiskReport",
    "NavigationState",
    "PathEvaluation",
    "Pose",
    "RuntimeMonitor",
    "RuntimeObservation",
    "SafeModeSupervisor",
    "SafetyMode",
    "SelfAwareCostWeights",
    "SelfAwarenessScore",
    "Trajectory",
    "UncertaintyPropagator",
    "UncertaintyState",
    "estimate_self_awareness",
    "expected_information_gain",
    "self_aware_path_cost",
    "__version__",
]
