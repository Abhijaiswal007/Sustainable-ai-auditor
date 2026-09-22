import json

from ollama import chat
from pydantic import BaseModel, Field

from requirements_parser import parse_requirements


OLLAMA_MODEL = "qwen3:1.7b"

DEFAULT_MIN_ACCURACY = 0.90
DEFAULT_MAX_LATENCY_MS = 200.0


class ExtractedRequirements(BaseModel):
    """
    Schema that Qwen3 must follow when extracting constraints.
    """

    min_accuracy: float | None = Field(
        default=None,
        ge=0,
        le=100
    )

    max_latency_ms: float | None = Field(
        default=None,
        ge=0
    )

    max_carbon_kgco2e: float | None = Field(
        default=None,
        ge=0
    )


def normalize_accuracy(value):
    """
    Convert 90 or 90% into 0.90.
    Keep 0.90 as 0.90.
    """

    if value is None:
        return DEFAULT_MIN_ACCURACY

    number = float(value)

    if number > 1:
        number = number / 100

    if not 0 <= number <= 1:
        raise ValueError(
            f"Local AI returned invalid accuracy: {value}"
        )

    return number


def parse_requirements_with_ai(description):
    """
    Qwen3 interprets the request, while the deterministic
    parser verifies and preserves the exact numeric values.
    """

    if not description or not description.strip():
        return parse_requirements(description)

    system_prompt = """
You extract constraints for a sustainable AI model-selection system.

Return only the fields required by the supplied JSON schema.

Rules:
1. Extract minimum accuracy.
2. Extract maximum latency in milliseconds.
3. Extract maximum carbon in kilograms of CO2e.
4. Use null when a constraint is not supplied.
5. Copy numbers exactly as written.
6. Never estimate, round, modify, or calculate numbers.
7. Do not recommend a model.
"""

    try:
        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": description
                }
            ],
            format=ExtractedRequirements.model_json_schema(),
            options={
                "temperature": 0
            }
        )

        # Validate that Qwen3 produced the required structure.
        ExtractedRequirements.model_validate_json(
            response.message.content
        )

    except Exception as error:
        raise RuntimeError(
            "The local Qwen3 requirement parser failed."
        ) from error

    # Numeric values come from the deterministic parser.
    # This prevents the language model from changing 150 to 149
    # or 60 to 59.999.
    verified_constraints = parse_requirements(description)

    return verified_constraints


def build_explanation_facts(decision):
    """
    Keep only verified facts that Qwen3 is allowed to explain.
    """

    constraints = decision["constraints"]

    facts = {
        "decision_status": decision["status"],
        "constraints": constraints,
        "selected_option": None,
        "eligible_options": [],
        "rejected_options": decision["rejected_options"]
    }

    if decision["selected"] is not None:
        selected = decision["selected"]

        facts["selected_option"] = {
            "option_id": selected["option_id"],
            "model": selected["model"],
            "region": selected["region"],
            "accuracy": selected["accuracy"],
            "latency_ms": selected["latency_ms"],
            "total_kgco2e": selected["total_kgco2e"],
            "low_kgco2e": selected["low_kgco2e"],
            "high_kgco2e": selected["high_kgco2e"]
        }

    for option in decision["eligible_options"]:
        facts["eligible_options"].append({
            "option_id": option["option_id"],
            "model": option["model"],
            "accuracy": option["accuracy"],
            "latency_ms": option["latency_ms"],
            "total_kgco2e": option["total_kgco2e"]
        })

    return facts


def generate_explanation_with_ai(decision):
    """
    Ask local Qwen3 to explain the deterministic decision.
    Qwen3 does not make or modify the decision.
    """

    facts = build_explanation_facts(decision)

    prompt = f"""
Write a short explanation of this sustainable AI recommendation.

Verified facts:
{json.dumps(facts, indent=2)}

Instructions:
- Use only the verified facts above.
- Do not invent measurements.
- Do not change the selected option.
- Explain why rejected options failed.
- Mention accuracy, latency, and lifecycle carbon.
- Use three to five short sentences.
- If no option is eligible, say so clearly.
"""

    try:
        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain verified sustainability decisions. "
                        "You must never modify or invent numeric results."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        explanation = response.message.content.strip()

        if not explanation:
            raise ValueError(
                "The local model returned an empty explanation."
            )

        return explanation

    except Exception as error:
        raise RuntimeError(
            "The local Qwen3 explanation generator failed."
        ) from error