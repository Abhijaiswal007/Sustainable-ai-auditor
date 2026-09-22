def evaluate_option(option, constraints):
    """
    Check whether one option satisfies all user constraints.
    """

    rejection_reasons = []

    min_accuracy = constraints["min_accuracy"]
    max_latency = constraints["max_latency_ms"]
    max_carbon = constraints.get("max_carbon_kgco2e")

    if option["accuracy"] < min_accuracy:
        rejection_reasons.append(
            f"accuracy {option['accuracy']:.1%} is below "
            f"the required {min_accuracy:.1%}"
        )

    if option["latency_ms"] > max_latency:
        rejection_reasons.append(
            f"latency {option['latency_ms']:.1f} ms exceeds "
            f"the maximum {max_latency:.1f} ms"
        )

    if (
        max_carbon is not None
        and option["total_kgco2e"] > max_carbon
    ):
        rejection_reasons.append(
            f"carbon {option['total_kgco2e']:.2f} kg CO2e "
            f"exceeds the maximum {max_carbon:.2f} kg CO2e"
        )

    return rejection_reasons


def recommend_option(options, constraints):
    """
    Reject options that violate constraints, then select the
    eligible option with the lowest lifecycle carbon.
    """

    if not options:
        raise ValueError("No model options were provided.")

    eligible_options = []
    rejected_options = []

    for option in options:
        rejection_reasons = evaluate_option(
            option,
            constraints
        )

        if rejection_reasons:
            rejected_options.append({
                "option_id": option["option_id"],
                "model": option["model"],
                "reasons": rejection_reasons
            })
        else:
            eligible_options.append(option)

    if not eligible_options:
        return {
            "status": "no_eligible_option",
            "selected": None,
            "eligible_options": [],
            "rejected_options": rejected_options,
            "constraints": constraints
        }

    # Primary rule: lowest carbon.
    # Tie-breaker 1: highest accuracy.
    # Tie-breaker 2: lowest latency.
    selected = min(
        eligible_options,
        key=lambda option: (
            option["total_kgco2e"],
            -option["accuracy"],
            option["latency_ms"]
        )
    )

    return {
        "status": "success",
        "selected": selected,
        "eligible_options": eligible_options,
        "rejected_options": rejected_options,
        "constraints": constraints
    }