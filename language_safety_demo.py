from language_safety_policy import LanguageSafetyConfig, LanguageSafetyPolicy


def main():
    config = LanguageSafetyConfig()
    policy = LanguageSafetyPolicy(config=config)

    base_risk = 1.0
    base_unc = 0.3

    messages = [
        "Εκεί έχει πολύ κόσμο, ειδικά στο διάδρομο.",
        "Πρόσεχε, ο διάδρομος είναι γλιστερός.",
        "There are stairs ahead, and many elderly people.",
        "Balanced corridor, nothing special here.",
        "Πρόσεχε, δεν το βλέπεις αλλά στη γωνία έχει κόσμο και παιδιά.",
    ]

    print("=== Language-Driven Safety Demo ===")
    print(f"Base risk={base_risk}, base_uncertainty={base_unc}\n")

    for i, msg in enumerate(messages):
        result = policy.modulate_risk_and_uncertainty(
            base_risk=base_risk,
            base_uncertainty=base_unc,
            text=msg,
        )

        print(f"Message {i}: \"{msg}\"")
        print(f"  risk_scale        = {result['risk_scale']:.2f}")
        print(f"  uncertainty_scale = {result['uncertainty_scale']:.2f}")
        print(f"  new risk          = {result['risk']:.3f}")
        print(f"  new uncertainty   = {result['uncertainty']:.3f}")
        print(f"  explanation       = {result['explanation']}")
        print('  -> π.χ. ο planner αυξάνει λ ή αποφεύγει το συγκεκριμένο corridor.\n')


if __name__ == "__main__":
    main()
