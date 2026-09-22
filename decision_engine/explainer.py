def generate_explanation(decision):
    """
    Create a human-readable explanation without an AI API.
    """

    if decision["status"] == "no_eligible_option":
        rejected_lines = []

        for rejected in decision["rejected_options"]:
            reasons = "; ".join(rejected["reasons"])

            rejected_lines.append(
                f"{rejected['model']} was rejected because {reasons}."
            )

        rejection_text = " ".join(rejected_lines)

        return (
            "No model satisfies all the requested constraints. "
            + rejection_text
        )

    selected = decision["selected"]
    constraints = decision["constraints"]

    explanation = (
        f"We recommend {selected['model']} in "
        f"{selected['region']} because it satisfies the required "
        f"minimum accuracy of "
        f"{constraints['min_accuracy']:.1%} and maximum latency of "
        f"{constraints['max_latency_ms']:.1f} ms. "
        f"It achieves {selected['accuracy']:.1%} accuracy with "
        f"{selected['latency_ms']:.1f} ms latency. "
        f"Its estimated lifecycle footprint is "
        f"{selected['total_kgco2e']:.2f} kg CO2e, with a range of "
        f"{selected['low_kgco2e']:.2f} to "
        f"{selected['high_kgco2e']:.2f} kg CO2e. "
        f"Among all eligible options, it has the lowest expected "
        f"lifecycle carbon footprint."
    )

    if decision["rejected_options"]:
        rejected_names = [
            item["model"]
            for item in decision["rejected_options"]
        ]

        explanation += (
            " Rejected options were: "
            + ", ".join(rejected_names)
            + "."
        )

    return explanation