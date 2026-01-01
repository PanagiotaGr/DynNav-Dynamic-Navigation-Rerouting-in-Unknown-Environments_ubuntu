from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class LanguageSafetyConfig:
    """
    How verbal descriptions modulate risk / uncertainty.

    Multipliers are applied on top of base risk / uncertainty.
    """
    crowding_risk_scale: float = 1.8
    slippery_risk_scale: float = 2.0
    stairs_risk_scale: float = 1.5
    children_risk_scale: float = 1.7
    elderly_risk_scale: float = 1.7

    unseen_hazard_uncertainty_scale: float = 1.5

    max_risk_scale: float = 3.0
    max_uncertainty_scale: float = 3.0


@dataclass
class LanguageSafetyFactors:
    crowding: bool = False
    slippery: bool = False
    stairs: bool = False
    children: bool = False
    elderly: bool = False
    unseen_hazard: bool = False


class LanguageSafetyPolicy:
    """
    Parse natural language hints (EN/GR) and modulate risk/uncertainty.

    Example inputs:
      - "έχει πολύ κόσμο εκεί"
      - "the corridor is slippery"
      - "there are stairs ahead"
      - "many children and elderly people here"
    """

    def __init__(self, config: LanguageSafetyConfig | None = None):
        if config is None:
            config = LanguageSafetyConfig()
        self.config = config

    def parse_text(self, text: str) -> LanguageSafetyFactors:
        text_l = text.lower()

        f = LanguageSafetyFactors()

        # Crowding / many people
        if any(
            kw in text_l
            for kw in [
                "πολύ κόσμο",
                "πολυκοσμία",
                "έχει κόσμο",
                "crowded",
                "many people",
                "a lot of people",
            ]
        ):
            f.crowding = True

        # Slippery / wet floor
        if any(
            kw in text_l
            for kw in [
                "γλιστερ",
                "γλιστρά",
                "wet floor",
                "slippery",
                "slippery floor",
            ]
        ):
            f.slippery = True

        # Stairs
        if any(
            kw in text_l
            for kw in [
                "σκαλές",
                "σκάλες",
                "stairs",
                "staircase",
            ]
        ):
            f.stairs = True

        # Children
        if any(
            kw in text_l
            for kw in [
                "παιδιά",
                "μωρά",
                "children",
                "kids",
            ]
        ):
            f.children = True

        # Elderly
        if any(
            kw in text_l
            for kw in [
                "ηλικιωμένοι",
                "γηραιοί",
                "elderly",
                "old people",
            ]
        ):
            f.elderly = True

        # Unseen hazard / warnings
        if any(
            kw in text_l
            for kw in [
                "πρόσεχε",
                "προσοχή",
                "δεν το βλέπεις",
                "you can't see it",
                "hidden",
                "around the corner",
            ]
        ):
            f.unseen_hazard = True

        return f

    def modulate_risk_and_uncertainty(
        self,
        base_risk: float,
        base_uncertainty: float,
        text: str,
    ) -> Dict[str, Any]:
        """
        Apply language-driven modulation to risk and uncertainty.

        Returns a dict with:
          - risk
          - uncertainty
          - factors
          - explanation
        """
        factors = self.parse_text(text)

        risk_scale = 1.0
        unc_scale = 1.0
        explanations: List[str] = []

        if factors.crowding:
            risk_scale *= self.config.crowding_risk_scale
            explanations.append("crowding / many people")

        if factors.slippery:
            risk_scale *= self.config.slippery_risk_scale
            explanations.append("slippery / wet floor")

        if factors.stairs:
            risk_scale *= self.config.stairs_risk_scale
            explanations.append("stairs / elevation changes")

        if factors.children:
            risk_scale *= self.config.children_risk_scale
            explanations.append("children present")

        if factors.elderly:
            risk_scale *= self.config.elderly_risk_scale
            explanations.append("elderly people present")

        if factors.unseen_hazard:
            unc_scale *= self.config.unseen_hazard_uncertainty_scale
            explanations.append("unseen / warned hazard")

        # Clamp scales to avoid insane explosions
        risk_scale = min(risk_scale, self.config.max_risk_scale)
        unc_scale = min(unc_scale, self.config.max_uncertainty_scale)

        new_risk = base_risk * risk_scale
        new_unc = base_uncertainty * unc_scale

        explanation_text = (
            "No language-driven adjustments."
            if not explanations
            else "Language-driven factors: " + ", ".join(explanations)
        )

        return {
            "base_risk": base_risk,
            "base_uncertainty": base_uncertainty,
            "risk": new_risk,
            "uncertainty": new_unc,
            "risk_scale": risk_scale,
            "uncertainty_scale": unc_scale,
            "factors": factors,
            "explanation": explanation_text,
        }
