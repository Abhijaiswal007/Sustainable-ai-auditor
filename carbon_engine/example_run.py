import csv
import json
from pathlib import Path

from carbon_calculator import (
    calculate_all_scenarios,
    load_reference_data
)


BASE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = BASE_DIRECTORY / "data"
OUTPUT_DIRECTORY = BASE_DIRECTORY / "outputs"


# This represents one example AI system.
system_description = {
    "region": "India",
    "gpu_name": "NVIDIA T4",
    "gpu_count": 1,

    # Initial model training
    "training_hours": 8,
    "training_runs": 1,

    # Daily inference
    "energy_per_request_kwh": 0.000002,
    "requests_per_day": 100000,
    "inference_device_hours_per_day": 2,

    # Storage and networking
    "storage_gb": 50,
    "network_gb_per_month": 100,

    # Periodic retraining
    "retraining_hours_per_run": 2,
    "retraining_runs_per_year": 4,

    # Calculation period
    "days": 365,
    "months": 12
}


def create_summary_row(label, result):
    stages = result["stages"]

    return {
        "scenario": label,
        "region": result["region"],
        "gpu_name": result["gpu_name"],
        "training_kgco2e": stages["training"]["carbon_kgco2e"],
        "inference_kgco2e": stages["inference"]["carbon_kgco2e"],
        "storage_kgco2e": stages["storage"]["carbon_kgco2e"],
        "networking_kgco2e": stages["networking"]["carbon_kgco2e"],
        "retraining_kgco2e": stages["retraining"]["carbon_kgco2e"],
        "hardware_kgco2e": stages["hardware"]["carbon_kgco2e"],
        "total_energy_kwh": result["total_energy_kwh"],
        "total_kgco2e": result["total_kgco2e"]
    }


def main():
    reference_data = load_reference_data(DATA_DIRECTORY)

    results = calculate_all_scenarios(
        system=system_description,
        reference_data=reference_data
    )

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    json_path = OUTPUT_DIRECTORY / "lifecycle_results.json"
    csv_path = OUTPUT_DIRECTORY / "lifecycle_results.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    rows = [
        create_summary_row("low", results["low"]),
        create_summary_row("expected", results["expected"]),
        create_summary_row("high", results["high"])
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nLIFECYCLE CARBON ESTIMATE")
    print("=" * 45)

    for label, result in results.items():
        print(
            f"{label.capitalize():10} "
            f"{result['total_kgco2e']:.4f} kg CO2e"
        )

    print("\nStage breakdown for expected scenario:")

    expected_stages = results["expected"]["stages"]

    for stage_name, stage_result in expected_stages.items():
        carbon = stage_result["carbon_kgco2e"]
        print(f"  {stage_name:12}: {carbon:.4f} kg CO2e")

    print(f"\nJSON saved to: {json_path}")
    print(f"CSV saved to:  {csv_path}")


if __name__ == "__main__":
    main()