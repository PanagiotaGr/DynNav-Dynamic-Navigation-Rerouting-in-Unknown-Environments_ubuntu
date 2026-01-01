from self_healing_policy import SelfHealingConfig, SelfHealingPolicy


def main():
    config = SelfHealingConfig(
        drift_threshold=0.6,
        calib_error_threshold=0.5,
        regret_threshold=0.5,
        failure_rate_threshold=0.3,
        min_steps_between_actions=3,
    )
    policy = SelfHealingPolicy(config=config)

    # Synthetic time-series of metrics per step
    # (step, drift, calib_error, regret, failure_rate)
    trajectory = [
        (0, 0.2, 0.1, 0.1, 0.0),
        (1, 0.4, 0.2, 0.2, 0.1),
        (2, 0.65, 0.3, 0.2, 0.1),  # high drift
        (3, 0.7, 0.6, 0.3, 0.25),  # drift + calib
        (4, 0.5, 0.7, 0.6, 0.35),  # calib + regret + failures
        (7, 0.3, 0.4, 0.7, 0.4),   # mostly regret + failures
        (10, 0.7, 0.2, 0.2, 0.0),  # drift again
    ]

    print("=== Self-Healing Navigation Demo ===")
    for step, drift, calib, regret, fail_rate in trajectory:
        trigger, info = policy.should_self_heal(
            step=step,
            drift=drift,
            calib_error=calib,
            regret=regret,
            failure_rate=fail_rate,
        )

        print(f"\nStep {step}:")
        print(f"  drift        = {drift:.2f}")
        print(f"  calib_error  = {calib:.2f}")
        print(f"  regret       = {regret:.2f}")
        print(f"  failure_rate = {fail_rate:.2f}")
        if trigger:
            print("  SELF_HEALING_TRIGGER = True")
            print("  reasons:")
            for r in info["reasons"]:
                print(f"    - {r}")
            print("  recommended_actions:")
            for a in info["recommended_actions"]:
                print(f"    - {a}")
            print('  -> π.χ. "Επανεκπαιδεύω heuristic με νέο dataset και '
                  'ξανατρέχω calibration για uncertainty grids."')
        else:
            print("  SELF_HEALING_TRIGGER = False")


if __name__ == "__main__":
    main()
