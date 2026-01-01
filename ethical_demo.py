from ethical_risk_policy import EthicalRiskConfig, EthicalRiskPolicy
from ethical_zones import EthicalZoneManager
from risk_cost_utils import human_and_ethical_edge_cost


def main():
    # Example robot lambda for physical risk:
    lambda_robot = 1.0

    # Configure ethical penalties
    config = EthicalRiskConfig(
        ethical_weight=1.0,
        human_proximity_penalty=5.0,
        private_zone_penalty=8.0,
        vulnerable_area_penalty=10.0,
    )
    ethical_policy = EthicalRiskPolicy(config=config)

    # Load zones
    zone_manager = EthicalZoneManager("ethical_zones.json")

    # Three example edges (e.g. midpoints of candidate paths)
    edges = [
        {"name": "A_short_but_private", "x": 2.0, "y": 2.0,
         "base_length": 5.0, "physical_risk": 0.5, "near_human": False},
        {"name": "B_long_and_clean", "x": 0.0, "y": 0.0,
         "base_length": 8.0, "physical_risk": 0.3, "near_human": False},
        {"name": "C_near_human", "x": -1.0, "y": 4.0,
         "base_length": 6.0, "physical_risk": 0.4, "near_human": True},
    ]

    print("=== Ethical Navigation Demo ===")
    for e in edges:
        in_private, in_vulnerable = zone_manager.check_zone_status(e["x"], e["y"])

        costs = human_and_ethical_edge_cost(
            base_length=e["base_length"],
            physical_risk=e["physical_risk"],
            lambda_weight=lambda_robot,
            ethical_policy=ethical_policy,
            near_human=e["near_human"],
            in_private_zone=in_private,
            in_vulnerable_area=in_vulnerable,
        )

        print(f"\nPath {e['name']}:")
        print(f"  base_length        = {costs['base_length']:.3f}")
        print(f"  physical_risk      = {costs['physical_risk']:.3f}")
        print(f"  physical_risk_cost = {costs['physical_risk_cost']:.3f}")
        print(f"  ethical_risk       = {costs['ethical_risk']:.3f}")
        print(f"  lambda             = {costs['lambda_weight']:.3f}")
        print(f"  TOTAL COST         = {costs['total_cost']:.3f}")

    print("\nChoose the path with minimum TOTAL COST to see the ethical effect.")


if __name__ == "__main__":
    main()
