from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AskForHelpConfig:
    """
    Configuration for a simple ask-for-help policy.

    All signals are assumed normalized in [0, 1]:

      - uncertainty: 0 = perfectly certain, 1 = maximally uncertain
      - self_trust:  0 = no trust in self, 1 = full confidence
    """
    uncertainty_threshold: float = 0.7
    self_trust_threshold: float = 0.3
    min_steps_between_requests: int = 5
    enable_uncertainty_trigger: bool = True
    enable_trust_trigger: bool = True


class AskForHelpPolicy:
    """
    Implements a simple human-in-the-loop triggering policy.

    The robot "asks for help" when uncertainty is too high and/or
    when its self-trust is too low, with a refractory period
    (min_steps_between_requests) to avoid spam.
    """

    def __init__(self, config: Optional[AskForHelpConfig] = None):
        if config is None:
            config = AskForHelpConfig()
        self.config = config
        self._last_request_step: Optional[int] = None

    def reset(self):
        """Reset internal state (e.g., at the start of a new episode)."""
        self._last_request_step = None

    def _cooldown_ok(self, step: int) -> bool:
        if self._last_request_step is None:
            return True
        return (step - self._last_request_step) >= self.config.min_steps_between_requests

    def should_ask_for_help(
        self,
        step: int,
        uncertainty: float,
        self_trust: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide whether to ask for human help at this step.

        Parameters
        ----------
        step : int
            Current discrete time step / planning iteration.
        uncertainty : float in [0, 1]
            Current normalized uncertainty level.
        self_trust : float in [0, 1]
            Current normalized self-trust level.

        Returns
        -------
        ask : bool
            Whether to trigger a help request.
        reason : Optional[str]
            Textual explanation for logging / human display.
        """
        if not self._cooldown_ok(step):
            return False, None

        reasons = []

        if self.config.enable_uncertainty_trigger and uncertainty >= self.config.uncertainty_threshold:
            reasons.append(
                f"uncertainty {uncertainty:.2f} ≥ threshold {self.config.uncertainty_threshold:.2f}"
            )

        if self.config.enable_trust_trigger and self_trust <= self.config.self_trust_threshold:
            reasons.append(
                f"self_trust {self_trust:.2f} ≤ threshold {self.config.self_trust_threshold:.2f}"
            )

        if not reasons:
            return False, None

        # If we reach here, at least one condition fired
        self._last_request_step = step

        # Combine reasons into a concise explanation.
        reason_text = " and ".join(reasons)
        return True, f"Requesting help because {reason_text}."
