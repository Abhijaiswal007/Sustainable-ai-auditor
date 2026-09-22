import csv
from pathlib import Path

from carbon_calculator import calculate_all_scenarios, load_reference_data


CARBON_ENGINE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = CARBON_ENGINE_DIRECTORY.parent

REFERENCE_DATA_DIRECTORY = CARBON_ENGINE_DIRECTORY / "data"
ROLE2_INPUT_FILE = (
    PROJECT_DIRECTORY
    / "role2_experiments"
    / "results"
    / "role1_input.csv"
)
OUTPUT_FILE = (
    CARBON_ENGINE_DIRECTORY
    / "outputs"
    / "model_carbon_results.csv"
)


# Shared deployment assumptions. These are editable workload inputs,
# not precomputed carbon results. Role 2 supplies the model-specific
# measured energy per request.
COMMON_SYSTEM = {
    "region": "India",
    "gpu_name": "NVIDIA T4",
    "gpu_count": 1,
    "training_hours": 8,
    "training_runs": 1,
    "requests_per_day": 100000,
    "inference_device_hours_per_day": 2,
    "storage_gb": 50,
    "network_gb_per_month": 100,
    "retraining_hours_per_run": 2,
    "retraining_runs_per_year": 4,
    "days": 365,
    "months": 12,
}


ROLE2_REQUIRED_COLUMNS = {
    "experiment_id",
    "option_id",
    "model_name",
    "energy_per_request_kwh",
    "measurement_device",
    "measurement_sample_count",
    "runs",
}


OUTPUT_COLUMNS = [
    "experiment_id",
    "option_id",
    "model",
    "region",
    "measurement_device",
    "measurement_sample_count",
    "runs",
    "training_kgco2e",
    "inference_kgco2e",
    "storage_kgco2e",
    "networking_kgco2e",
    "retraining_kgco2e",
    "hardware_kgco2e",
    "total_kgco2e",
    "low_kgco2e",
    "high_kgco2e",
]


def read_role2_measurements(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Role 2 input file was not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        actual_columns = set(reader.fieldnames or [])
        missing_columns = ROLE2_REQUIRED_COLUMNS - actual_columns

        if missing_columns:
            raise ValueError(
                "Role 2 input is missing columns: "
                f"{sorted(missing_columns)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Role 2 input contains no model rows.")

    seen_option_ids = set()

    for row in rows:
        option_id = row["option_id"].strip()

        if not option_id:
            raise ValueError("Role 2 input contains an empty option_id.")

        if option_id in seen_option_ids:
            raise ValueError(f"Duplicate option_id: {option_id}")

        seen_option_ids.add(option_id)

        try:
            energy = float(row["energy_per_request_kwh"])
        except ValueError as error:
            raise ValueError(
                f"Energy per request for {option_id} must be numeric."
            ) from error

        if energy < 0:
            raise ValueError(
                f"Energy per request for {option_id} cannot be negative."
            )

    return rows


def create_output_row(measurement, results):
    expected = results["expected"]
    stages = expected["stages"]

    return {
        "experiment_id": measurement["experiment_id"].strip(),
        "option_id": measurement["option_id"].strip(),
        "model": measurement["model_name"].strip(),
        "region": expected["region"],
        "measurement_device": measurement["measurement_device"].strip(),
        "measurement_sample_count": measurement[
            "measurement_sample_count"
        ],
        "runs": measurement["runs"],
        "training_kgco2e": stages["training"]["carbon_kgco2e"],
        "inference_kgco2e": stages["inference"]["carbon_kgco2e"],
        "storage_kgco2e": stages["storage"]["carbon_kgco2e"],
        "networking_kgco2e": stages["networking"]["carbon_kgco2e"],
        "retraining_kgco2e": stages["retraining"]["carbon_kgco2e"],
        "hardware_kgco2e": stages["hardware"]["carbon_kgco2e"],
        "total_kgco2e": expected["total_kgco2e"],
        "low_kgco2e": results["low"]["total_kgco2e"],
        "high_kgco2e": results["high"]["total_kgco2e"],
    }


def main():
    measurements = read_role2_measurements(ROLE2_INPUT_FILE)
    reference_data = load_reference_data(REFERENCE_DATA_DIRECTORY)
    output_rows = []

    for measurement in measurements:
        system = COMMON_SYSTEM.copy()
        system["energy_per_request_kwh"] = float(
            measurement["energy_per_request_kwh"]
        )

        results = calculate_all_scenarios(
            system=system,
            reference_data=reference_data,
        )

        output_rows.append(create_output_row(measurement, results))

    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print("MODEL LIFECYCLE CARBON RESULTS")
    print("=" * 70)

    for row in output_rows:
        print(
            f"{row['option_id']} | {row['model']} | "
            f"Expected: {float(row['total_kgco2e']):.4f} kg CO2e | "
            f"Range: {float(row['low_kgco2e']):.4f}-"
            f"{float(row['high_kgco2e']):.4f} kg CO2e"
        )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
