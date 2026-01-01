from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List


@dataclass
class SelfHealingConfig:
    """
    Configuration thresholds for self-healing navigation.

    All metrics are assumed normalized or in comparable units:

      - drift:              0 (no drift) .. 1 (very high drift)
      - calib_error:        0 (well calibrated) .. 1 (badly calibrated)
      - regret:             0 (perfect heuristic) .. 1 (high regret)
      - failure_rate:       0 .. 1 over a recent window
    """
    drift_threshold: float = 0.6
    calib_error_threshold: float = 0.5
    regret_threshold: float = 0.5
    failure_rate_threshold: float = 0.3

    min_steps_between_actions: int = 10

    enable_drift_trigger: bool = True
    enable_calib_trigger: bool = True
    enable_regret_trigger: bool = True
    enable_failure_trigger: bool = True


class SelfHealingPolicy:
    """
    Self-healing controller for navigation.

    Given current metrics (drift, calibration error, heuristic regret, failure rate),
    it suggests when the system should "repair itself", e.g.:

      - re-calibrate uncertainty estimator
      - re-train learned heuristic (small online batch)
      - increase risk aversion (lambda)
      - temporarily force safe mode
    """

    def __init__(self, config: Optional[SelfHealingConfig] = None):
        if config is None:
            config = SelfHealingConfig()
        self.config = config
        self._last_action_step: Optional[int] = None

    def reset(self):
        self._last_action_step = None

    def _cooldown_ok(self, step: int) -> bool:
        if self._last_action_step is None:
            return True
        return (step - self._last_action_step) >= self.config.min_steps_between_actions

    def should_self_heal(
        self,
        step: int,
        drift: float,
        calib_error: float,
        regret: float,
        failure_rate: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Decide whether to trigger self-healing behavior.

        Parameters
        ----------
        step : int
            Current time step.
        drift : float
            Normalized drift metric.
        calib_error : float
            Normalized calibration error metric.
        regret : float
            Normalized heuristic regret (e.g. extra cost vs optimal).
        failure_rate : float
            Recent failure / near-miss rate over a sliding window.

        Returns
        -------
        trigger : bool
            Whether to recommend self-healing actions.
        info : dict
            Contains 'reasons' and 'recommended_actions'.
        """
        if not self._cooldown_ok(step):
            return False, {"reasons": [], "recommended_actions": []}

        reasons: List[str] = []
        actions: List[str] = []

        # Drift too high → consider changing estimator / fusion parameters
        if self.config.enable_drift_trigger and drift >= self.config.drift_threshold:
            reasons.append(
                f"drift {drift:.2f} ≥ threshold {self.config.drift_threshold:.2f}"
            )
            actions.append("adjust_state_estimator")  # e.g. re-tune UKF / VO fusion
            actions.append("increase_risk_weight")    # more conservative paths

        # Calibration error high → rebuild / recalibrate uncertainty grids
        if self.config.enable_calib_trigger and calib_error >= self.config.calib_error_threshold:
            reasons.append(
                f"calibration_error {calib_error:.2f} ≥ threshold {self.config.calib_error_threshold:.2f}"
            )
            actions.append("recalibrate_uncertainty_model")  # re-run calibration scripts

        # Heuristic regret high → on-the-fly small-batch re-training
        if self.config.enable_regret_trigger and regret >= self.config.regret_threshold:
            reasons.append(
                f"regret {regret:.2f} ≥ threshold {self.config.regret_threshold:.2f}"
            )
            actions.append("retrain_learned_heuristic_small_batch")

        # Failure / near-miss rate high → safe mode + strong risk aversion
        if self.config.enable_failure_trigger and failure_rate >= self.config.failure_rate_threshold:
            reasons.append(
                f"failure_rate {failure_rate:.2f} ≥ threshold {self.config.failure_rate_threshold:.2f}"
            )
            actions.append("enter_safe_mode")
            if "increase_risk_weight" not in actions:
                actions.append("increase_risk_weight")

        if not reasons:
            return False, {"reasons": [], "recommended_actions": []}

        actions = sorted(set(actions))
        self._last_action_step = step

        return True, {
            "reasons": reasons,
            "recommended_actions": actions,
        }
