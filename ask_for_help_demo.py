from ask_for_help_policy import AskForHelpConfig, AskForHelpPolicy


def main():
    config = AskForHelpConfig(
        uncertainty_threshold=0.7,
        self_trust_threshold=0.3,
        min_steps_between_requests=3,
    )
    policy = AskForHelpPolicy(config=config)

    # Example time series of (uncertainty, self_trust)
    # Think of it as one navigation episode with 12 steps.
    trajectory = [
        # step, uncertainty, self_trust
        (0, 0.2, 0.9),   # confident, low uncertainty
        (1, 0.4, 0.8),
        (2, 0.6, 0.6),
        (3, 0.8, 0.5),   # high uncertainty → candidate trigger
        (4, 0.85, 0.45), # still high, but cooldown may block spam
        (5, 0.9, 0.35),
        (6, 0.4, 0.25),  # low self-trust → candidate trigger
        (7, 0.5, 0.2),
        (8, 0.6, 0.3),
        (9, 0.75, 0.28), # both bad
        (10, 0.3, 0.5),
        (11, 0.2, 0.6),
    ]

    print("=== Ask-for-Help Policy Demo ===")
    print(
        f"uncertainty_threshold={config.uncertainty_threshold}, "
        f"self_trust_threshold={config.self_trust_threshold}, "
        f"min_steps_between_requests={config.min_steps_between_requests}"
    )

    for step, unc, trust in trajectory:
        ask, reason = policy.should_ask_for_help(
            step=step,
            uncertainty=unc,
            self_trust=trust,
        )

        print(f"\nStep {step}:")
        print(f"  uncertainty = {unc:.2f}")
        print(f"  self_trust  = {trust:.2f}")
        if ask:
            print(f"  ASK_FOR_HELP = True")
            print(f"  reason       = {reason}")
            print('  -> e.g. "Δεν είμαι σίγουρο ότι ο δρόμος είναι ασφαλής. Θες να συνεχίσω;"')
        else:
            print(f"  ASK_FOR_HELP = False")


if __name__ == "__main__":
    main()
