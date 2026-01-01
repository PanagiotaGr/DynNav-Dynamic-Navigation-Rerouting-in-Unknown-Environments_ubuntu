"""
risk_cost_utils.py

Utility συναρτήσεις για cost σε risk-aware navigation με ανθρώπινες προτιμήσεις.

Κύρια συνάρτηση:

    human_aware_edge_cost(
        base_length,
        edge_risk,
        lambda_effective,
        human_pref,
        is_dark=False,
        is_low_feature=False,
        low_feature_penalty=5.0,
        dark_area_penalty=5.0,
    )

Αυτή υλοποιεί:

    cost = base_length + lambda_effective * edge_risk
           + penalties(αν ο άνθρωπος θέλει να αποφεύγει σκοτεινές / low-feature περιοχές)
"""

from user_preferences import HumanPreference
from ethical_risk_policy import EthicalRiskPolicy, EthicalRiskConfig

def human_aware_edge_cost(
    base_length: float,
    edge_risk: float,
    lambda_effective: float,
    human_pref: HumanPreference,
    is_dark: bool = False,
    is_low_feature: bool = False,
    low_feature_penalty: float = 5.0,
    dark_area_penalty: float = 5.0,
) -> float:
    """
    Υπολογίζει το κόστος ενός edge / segment με βάση:

        base_length           : γεωμετρικό μήκος / χρόνο / energy
        edge_risk             : κάποιο risk metric (π.χ. κοντά σε εμπόδια, μεγάλη αβεβαιότητα)
        lambda_effective      : risk weight μετά τον συνδυασμό robot + human
        human_pref            : αντικείμενο HumanPreference
        is_dark               : αν η ακμή περνάει από σκοτεινή περιοχή
        is_low_feature        : αν η ακμή περνάει από low-feature περιοχή
        low_feature_penalty   : penalty για low-feature περιοχές (αν το ζητά ο άνθρωπος)
        dark_area_penalty     : penalty για σκοτεινές περιοχές (αν το ζητά ο άνθρωπος)

    Επιστρέφει:
        cost = base_length + lambda_effective * edge_risk + penalties
    """
    cost = base_length + lambda_effective * edge_risk

    if human_pref.avoid_dark_areas and is_dark:
        cost += dark_area_penalty

    if human_pref.avoid_low_feature_areas and is_low_feature:
        cost += low_feature_penalty

    return cost
def human_and_ethical_edge_cost(
    base_length: float,
    physical_risk: float,
    lambda_weight: float,
    ethical_policy: EthicalRiskPolicy,
    near_human: bool = False,
    in_private_zone: bool = False,
    in_vulnerable_area: bool = False,
) -> dict:
    """
    Compute a combined cost that includes:
      - base geometric length
      - physical risk scaled by lambda_weight
      - ethical risk as defined by ethical_policy

    Returns a dict so it is easy to log all components.
    """
    # Physical risk cost term (lambda * R_phys)
    physical_risk_cost = lambda_weight * physical_risk

    total_cost, ethical_risk = ethical_policy.edge_total_cost(
        base_length=base_length,
        physical_risk_cost=physical_risk_cost,
        near_human=near_human,
        in_private_zone=in_private_zone,
        in_vulnerable_area=in_vulnerable_area,
    )

    return {
        "total_cost": total_cost,
        "base_length": base_length,
        "physical_risk": physical_risk,
        "physical_risk_cost": physical_risk_cost,
        "ethical_risk": ethical_risk,
        "lambda_weight": lambda_weight,
    }
