from dataclasses import dataclass


@dataclass
class EthicalRiskConfig:
    """
    Configuration for ethical / moral navigation costs.
    All penalties are in "cost units" compatible with path length and risk.
    """
    ethical_weight: float = 1.0
    human_proximity_penalty: float = 5.0
    private_zone_penalty: float = 8.0
    vulnerable_area_penalty: float = 10.0


class EthicalRiskPolicy:
    """
    Computes ethical risk and associated cost terms for navigation edges.

    This policy is planner-agnostic: it does not assume A*, PRM, RRT, etc.
    It just takes metadata about a candidate edge (near humans, private zones,
    vulnerable areas) and returns an ethical risk and cost contribution.
    """

    def __init__(self, config: EthicalRiskConfig | None = None):
        if config is None:
            config = EthicalRiskConfig()
        self.config = config

    def compute_ethical_risk(
        self,
        near_human: bool = False,
        in_private_zone: bool = False,
        in_vulnerable_area: bool = False,
    ) -> float:
        """
        Compute a scalar ethical risk score for a given edge / cell.
        """
        risk = 0.0

        if near_human:
            risk += self.config.human_proximity_penalty

        if in_private_zone:
            risk += self.config.private_zone_penalty

        if in_vulnerable_area:
            risk += self.config.vulnerable_area_penalty

        return risk

    def edge_total_cost(
        self,
        base_length: float,
        physical_risk_cost: float,
        near_human: bool = False,
        in_private_zone: bool = False,
        in_vulnerable_area: bool = False,
    ) -> tuple[float, float]:
        """
        Combine physical risk cost with ethical risk penalties.

        Parameters
        ----------
        base_length : float
            Nominal edge length / time / energy.
        physical_risk_cost : float
            Already-weighted physical risk term, e.g. lambda_phys * R_phys.
        near_human, in_private_zone, in_vulnerable_area : bool
            Binary flags describing ethical context of the edge.

        Returns
        -------
        total_cost : float
            Combined cost L + lambda_phys * R_phys + beta * R_ethical.
        ethical_risk : float
            The unweighted ethical risk term R_ethical (for logging / analysis).
        """
        ethical_risk = self.compute_ethical_risk(
            near_human=near_human,
            in_private_zone=in_private_zone,
            in_vulnerable_area=in_vulnerable_area,
        )

        total_cost = (
            base_length
            + physical_risk_cost
            + self.config.ethical_weight * ethical_risk
        )

        return total_cost, ethical_risk
