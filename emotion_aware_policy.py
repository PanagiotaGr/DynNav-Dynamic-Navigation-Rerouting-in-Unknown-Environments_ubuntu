from dataclasses import dataclass
from typing import Literal, Optional, Tuple


EmotionState = Literal["relaxed", "neutral", "anxious", "panicked"]


@dataclass
class EmotionAwareConfig:
    """
    Configuration for emotion-aware risk modulation.

    The idea:
      - relaxed  → can afford lower risk weight (more efficiency)
      - neutral  → baseline behavior
      - anxious  → increase risk weight (safer)
      - panicked → very conservative + possibly force safe mode
    """
    # Multipliers applied to the robot's base lambda.
    relaxed_scale: float = 0.7
    neutral_scale: float = 1.0
    anxious_scale: float = 1.5
    panicked_scale: float = 2.0

    # Whether extreme anxiety should force safe mode.
    panicked_forces_safe_mode: bool = True
    anxious_may_trigger_safe_mode: bool = False


class EmotionAwarePolicy:
    """
    Maps human emotional state to:
      - an effective risk weight (lambda_effective)
      - a possible safe_mode override.

    This is agnostic to how emotion is estimated:
      - could be from questionnaires,
      - physiological signals,
      - voice analysis,
      - etc.
    """

    def __init__(self, config: Optional[EmotionAwareConfig] = None):
        if config is None:
            config = EmotionAwareConfig()
        self.config = config

    def emotion_to_scale(self, emotion: EmotionState) -> float:
        if emotion == "relaxed":
            return self.config.relaxed_scale
        if emotion == "neutral":
            return self.config.neutral_scale
        if emotion == "anxious":
            return self.config.anxious_scale
        if emotion == "panicked":
            return self.config.panicked_scale
        # Should never happen with proper typing, but keep a fallback:
        return self.config.neutral_scale

    def compute_lambda_and_safe_mode(
        self,
        lambda_robot: float,
        emotion: EmotionState,
    ) -> Tuple[float, bool]:
        """
        Given the robot's base lambda and the human emotional state,
        compute an emotion-aware lambda_effective and whether safe mode
        should be forced.

        Parameters
        ----------
        lambda_robot : float
            Base risk weight from robot's self-trust / planning policy.
        emotion : EmotionState
            Human emotional state: "relaxed", "neutral", "anxious", "panicked".

        Returns
        -------
        lambda_effective : float
            Emotion-modulated risk weight.
        force_safe_mode : bool
            If True, the navigation layer should enter safe mode regardless
            of trust, uncertainty, etc.
        """
        scale = self.emotion_to_scale(emotion)
        lambda_effective = lambda_robot * scale

        force_safe_mode = False
        if emotion == "panicked" and self.config.panicked_forces_safe_mode:
            force_safe_mode = True
        elif emotion == "anxious" and self.config.anxious_may_trigger_safe_mode:
            force_safe_mode = True

        return lambda_effective, force_safe_mode
