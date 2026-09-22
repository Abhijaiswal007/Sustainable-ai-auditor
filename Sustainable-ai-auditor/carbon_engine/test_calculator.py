from pathlib import Path

import pytest

from carbon_calculator import (
    calculate_all_scenarios,
    carbon_from_energy,
    inference_carbon,
    load_reference_data,
    training_carbon
)


BASE_DIRECTORY = Path(__file__).resolve().parent


def test_energy_to_carbon_conversion():
    result = carbon_from_energy(
        energy_kwh=10,
        grid_intensity_g_per_kwh=500
    )

    assert result == pytest.approx(5.0)


def test_training_calculation():
    result = training_carbon(
        power_w=100,
        hours=10,
        grid_intensity_g_per_kwh=500,
        gpu_count=1,
        utilization=1,
        pue=1
    )

    assert result["energy_kwh"] == pytest.approx(1.0)
    assert result["carbon_kgco2e"] == pytest.approx(0.5)


def test_zero_training_hours():
    result = training_carbon(
        power_w=100,
        hours=0,
        grid_intensity_g_per_kwh=500
    )

    assert result["energy_kwh"] == 0
    assert result["carbon_kgco2e"] == 0


def test_doubling_requests_doubles_inference():
    first = inference_carbon(
        energy_per_request_kwh=0.000002,
        requests_per_day=100,
        days=365,
        grid_intensity_g_per_kwh=700,
        pue=1.2
    )

    second = inference_carbon(
        energy_per_request_kwh=0.000002,
        requests_per_day=200,
        days=365,
        grid_intensity_g_per_kwh=700,
        pue=1.2
    )

    assert second["carbon_kgco2e"] == pytest.approx(
        first["carbon_kgco2e"] * 2
    )


def test_negative_input_is_rejected():
    with pytest.raises(ValueError):
        training_carbon(
            power_w=-100,
            hours=10,
            grid_intensity_g_per_kwh=500
        )


def test_scenario_order():
    reference_data = load_reference_data(
        BASE_DIRECTORY / "data"
    )

    system = {
        "region": "India",
        "gpu_name": "NVIDIA T4",
        "gpu_count": 1,
        "training_hours": 8,
        "training_runs": 1,
        "energy_per_request_kwh": 0.000002,
        "requests_per_day": 100000,
        "inference_device_hours_per_day": 2,
        "storage_gb": 50,
        "network_gb_per_month": 100,
        "retraining_hours_per_run": 2,
        "retraining_runs_per_year": 4,
        "days": 365,
        "months": 12
    }

    results = calculate_all_scenarios(
        system,
        reference_data
    )

    assert (
        results["low"]["total_kgco2e"]
        <= results["expected"]["total_kgco2e"]
        <= results["high"]["total_kgco2e"]
    )