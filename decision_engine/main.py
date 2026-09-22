import json
from pathlib import Path

from data_loader import load_and_merge_options
from explainer import generate_explanation
from local_ai import (
    generate_explanation_with_ai,
    parse_requirements_with_ai
)
from recommender import recommend_option
from requirements_parser import parse_requirements


BASE_DIRECTORY = Path(__file__).resolve().parent

DATA_DIRECTORY = BASE_DIRECTORY / "data"
OUTPUT_DIRECTORY = BASE_DIRECTORY / "outputs"

ROLE1_FILE = DATA_DIRECTORY / "role1_carbon.csv"
ROLE2_FILE = DATA_DIRECTORY / "role2_metrics.csv"

OUTPUT_FILE = OUTPUT_DIRECTORY / "recommendation.json"


def display_options(options):
    print("\nAVAILABLE OPTIONS")
    print("=" * 70)

    for option in options:
        print(
            f"{option['option_id']} | "
            f"{option['model']} | "
            f"Accuracy: {option['accuracy']:.1%} | "
            f"Latency: {option['latency_ms']:.1f} ms | "
            f"Carbon: {option['total_kgco2e']:.2f} kg CO2e"
        )


def display_constraints(constraints, parser_source):
    print("\nINTERPRETED CONSTRAINTS")
    print("=" * 70)
    print(f"Parser: {parser_source}")

    print(
        f"Minimum accuracy: "
        f"{constraints['min_accuracy']:.1%}"
    )

    print(
        f"Maximum latency: "
        f"{constraints['max_latency_ms']:.1f} ms"
    )

    max_carbon = constraints["max_carbon_kgco2e"]

    if max_carbon is None:
        print("Maximum carbon: Not specified")
    else:
        print(
            f"Maximum carbon: "
            f"{max_carbon:.2f} kg CO2e"
        )


def main():
    print("\nSUSTAINABLE AI DECISION ENGINE")
    print("=" * 70)

    options = load_and_merge_options(
        ROLE1_FILE,
        ROLE2_FILE
    )

    display_options(options)

    print(
        "\nDescribe your requirements."
        "\nExample: I need at least 90% accuracy "
        "and latency below 200 ms."
    )

    description = input("\nRequirements: ").strip()

    # First attempt: local Qwen3.
    # Backup: the original rule-based parser.
    try:
        constraints = parse_requirements_with_ai(
            description
        )

        parser_source = "Local Qwen3 AI + numeric validation"

    except Exception as error:
        print(
            "\nWarning: Local AI parser was unavailable."
        )
        print(f"Reason: {error}")
        print("Using rule-based parser instead.")

        constraints = parse_requirements(
            description
        )

        parser_source = "Rule-based fallback"

    display_constraints(
        constraints,
        parser_source
    )

    # The deterministic engine always makes the final decision.
    decision = recommend_option(
        options,
        constraints
    )

    # First attempt: local Qwen3 explanation.
    # Backup: the original template explanation.
    try:
        explanation = generate_explanation_with_ai(
            decision
        )

        explanation_source = "Local Qwen3 AI"

    except Exception as error:
        print(
            "\nWarning: Local AI explanation was unavailable."
        )
        print(f"Reason: {error}")
        print("Using verified template explanation instead.")

        explanation = generate_explanation(
            decision
        )

        explanation_source = "Template fallback"

    decision["original_request"] = description

    decision["processing"] = {
        "requirements_parser": parser_source,
        "decision_method": (
            "Deterministic constraint filtering "
            "and minimum-carbon selection"
        ),
        "explanation_generator": explanation_source
    }

    decision["explanation"] = explanation

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            decision,
            file,
            indent=4
        )

    print("\nDECISION")
    print("=" * 70)
    print(f"Decision method: Deterministic optimizer")
    print(f"Explanation source: {explanation_source}")
    print()
    print(explanation)

    if decision["selected"] is not None:
        selected = decision["selected"]

        print("\nSELECTED OPTION")
        print("=" * 70)
        print(f"Option: {selected['option_id']}")
        print(f"Model: {selected['model']}")
        print(f"Region: {selected['region']}")
        print(f"Accuracy: {selected['accuracy']:.1%}")
        print(f"Latency: {selected['latency_ms']:.1f} ms")

        print(
            f"Expected carbon: "
            f"{selected['total_kgco2e']:.2f} kg CO2e"
        )

        print(
            f"Carbon range: "
            f"{selected['low_kgco2e']:.2f}–"
            f"{selected['high_kgco2e']:.2f} kg CO2e"
        )

    print(f"\nResult saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()