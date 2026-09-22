import pytest

from explainer import generate_explanation
from recommender import recommend_option
from requirements_parser import parse_requirements


SAMPLE_OPTIONS = [
    {
        "option_id": "OPT-A",
        "model": "DistilBERT",
        "region": "India",
        "accuracy": 0.91,
        "latency_ms": 85.0,
        "total_kgco2e": 40.0,
        "low_kgco2e": 32.0,
        "high_kgco2e": 50.0
    },
    {
        "option_id": "OPT-B",
        "model": "BERT-base",
        "region": "India",
        "accuracy": 0.94,
        "latency_ms": 160.0,
        "total_kgco2e": 79.0,
        "low_kgco2e": 65.0,
        "high_kgco2e": 95.0
    },
    {
        "option_id": "OPT-C",
        "model": "MiniLM",
        "region": "India",
        "accuracy": 0.89,
        "latency_ms": 55.0,
        "total_kgco2e": 30.0,
        "low_kgco2e": 24.0,
        "high_kgco2e": 38.0
    }
]


def test_parser_reads_percentage_accuracy():
    constraints = parse_requirements(
        "I need accuracy of at least 92% "
        "and latency below 150 ms."
    )

    assert constraints["min_accuracy"] == pytest.approx(0.92)
    assert constraints["max_latency_ms"] == pytest.approx(150)


def test_parser_uses_defaults_for_empty_text():
    constraints = parse_requirements("")

    assert constraints["min_accuracy"] == pytest.approx(0.90)
    assert constraints["max_latency_ms"] == pytest.approx(200)


def test_recommender_selects_lowest_carbon_eligible_option():
    constraints = {
        "min_accuracy": 0.90,
        "max_latency_ms": 200,
        "max_carbon_kgco2e": None
    }

    decision = recommend_option(
        SAMPLE_OPTIONS,
        constraints
    )

    assert decision["status"] == "success"
    assert decision["selected"]["option_id"] == "OPT-A"


def test_low_accuracy_model_is_rejected():
    constraints = {
        "min_accuracy": 0.90,
        "max_latency_ms": 200,
        "max_carbon_kgco2e": None
    }

    decision = recommend_option(
        SAMPLE_OPTIONS,
        constraints
    )

    rejected_ids = {
        option["option_id"]
        for option in decision["rejected_options"]
    }

    assert "OPT-C" in rejected_ids


def test_no_option_satisfies_constraints():
    constraints = {
        "min_accuracy": 0.99,
        "max_latency_ms": 20,
        "max_carbon_kgco2e": None
    }

    decision = recommend_option(
        SAMPLE_OPTIONS,
        constraints
    )

    assert decision["status"] == "no_eligible_option"
    assert decision["selected"] is None


def test_explanation_contains_selected_model():
    constraints = {
        "min_accuracy": 0.90,
        "max_latency_ms": 200,
        "max_carbon_kgco2e": None
    }

    decision = recommend_option(
        SAMPLE_OPTIONS,
        constraints
    )

    explanation = generate_explanation(decision)

    assert "DistilBERT" in explanation
    assert "40.00 kg CO2e" in explanation

def test_latency_is_not_mistaken_for_accuracy():
    constraints = parse_requirements(
        "I need latency below 200 ms."
    )

    assert constraints["min_accuracy"] == pytest.approx(0.90)
    assert constraints["max_latency_ms"] == pytest.approx(200)


def test_accuracy_after_latency_is_parsed_correctly():
    constraints = parse_requirements(
        "Keep response time under 200 ms "
        "and accuracy at least 90%."
    )

    assert constraints["min_accuracy"] == pytest.approx(0.90)
    assert constraints["max_latency_ms"] == pytest.approx(200)