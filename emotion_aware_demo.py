from emotion_aware_policy import EmotionAwareConfig, EmotionAwarePolicy, EmotionState


def main():
    # Suppose the robot has internally chosen this lambda
    # from risk / trust / calibration logic:
    lambda_robot = 1.0

    config = EmotionAwareConfig(
        relaxed_scale=0.7,
        neutral_scale=1.0,
        anxious_scale=1.5,
        panicked_scale=2.0,
        panicked_forces_safe_mode=True,
        anxious_may_trigger_safe_mode=False,
    )
    policy = EmotionAwarePolicy(config=config)

    emotion_sequence: list[EmotionState] = [
        "relaxed",
        "neutral",
        "anxious",
        "panicked",
        "neutral",
    ]

    print("=== Emotion-Aware Risk Modulation Demo ===")
    print(f"Base lambda_robot = {lambda_robot:.2f}")
    print(
        f"Scales: relaxed={config.relaxed_scale}, neutral={config.neutral_scale}, "
        f"anxious={config.anxious_scale}, panicked={config.panicked_scale}"
    )

    for step, emotion in enumerate(emotion_sequence):
        lambda_eff, force_safe = policy.compute_lambda_and_safe_mode(
            lambda_robot=lambda_robot,
            emotion=emotion,
        )

        print(f"\nStep {step}:")
        print(f"  human_emotion     = {emotion}")
        print(f"  lambda_effective  = {lambda_eff:.2f}")
        print(f"  force_safe_mode   = {force_safe}")

        if force_safe:
            print('  -> SAFE MODE: "Ο χρήστης φαίνεται αγχωμένος/πανικοβλημένος, '
                  'επιλέγω πιο ασφαλή πορεία."')
        elif emotion == "relaxed":
            print('  -> More efficiency allowed: "Ο χρήστης είναι ήρεμος, '
                  'μπορώ να δώσω λίγο βάρος στην ταχύτητα."')
        elif emotion == "anxious":
            print('  -> Risk bumped up: "Ο χρήστης είναι αγχωμένος, αυξάνω την '
                  'προσοχή στην ασφάλεια."')


if __name__ == "__main__":
    main()
