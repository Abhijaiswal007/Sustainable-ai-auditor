import csv
from pathlib import Path


ROLE1_REQUIRED_COLUMNS = {
    "option_id",
    "model",
    "region",
    "training_kgco2e",
    "inference_kgco2e",
    "storage_kgco2e",
    "networking_kgco2e",
    "retraining_kgco2e",
    "hardware_kgco2e",
    "total_kgco2e",
    "low_kgco2e",
    "high_kgco2e"
}


ROLE2_REQUIRED_COLUMNS = {
    "option_id",
    "model",
    "accuracy",
    "latency_ms",
    "measured_energy_kwh"
}


def read_csv_indexed(file_path, required_columns):
    """
    Read a CSV file and index every row using option_id.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        actual_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - actual_columns

        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        indexed_rows = {}

        for row in reader:
            option_id = row["option_id"].strip()

            if not option_id:
                raise ValueError(
                    f"{file_path.name} contains an empty option_id."
                )

            if option_id in indexed_rows:
                raise ValueError(
                    f"Duplicate option_id found: {option_id}"
                )

            indexed_rows[option_id] = row

    return indexed_rows


def to_nonnegative_float(value, field_name, option_id):
    """
    Convert a CSV value to a nonnegative floating-point number.
    """

    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(
            f"{field_name} for {option_id} must be numeric."
        ) from error

    if number < 0:
        raise ValueError(
            f"{field_name} for {option_id} cannot be negative."
        )

    return number


def load_and_merge_options(role1_file, role2_file):
    """
    Combine Role 1 carbon values with Role 2 model metrics.
    """

    role1_rows = read_csv_indexed(
        role1_file,
        ROLE1_REQUIRED_COLUMNS
    )

    role2_rows = read_csv_indexed(
        role2_file,
        ROLE2_REQUIRED_COLUMNS
    )

    role1_ids = set(role1_rows)
    role2_ids = set(role2_rows)

    if role1_ids != role2_ids:
        missing_from_role1 = role2_ids - role1_ids
        missing_from_role2 = role1_ids - role2_ids

        raise ValueError(
            "Role 1 and Role 2 option IDs do not match. "
            f"Missing from Role 1: {sorted(missing_from_role1)}. "
            f"Missing from Role 2: {sorted(missing_from_role2)}."
        )

    merged_options = []

    for option_id in sorted(role1_ids):
        carbon_row = role1_rows[option_id]
        metrics_row = role2_rows[option_id]

        role1_model = carbon_row["model"].strip()
        role2_model = metrics_row["model"].strip()

        if role1_model != role2_model:
            raise ValueError(
                f"Model mismatch for {option_id}: "
                f"Role 1 has '{role1_model}', "
                f"but Role 2 has '{role2_model}'."
            )

        option = {
            "option_id": option_id,
            "model": role1_model,
            "region": carbon_row["region"].strip(),

            "accuracy": to_nonnegative_float(
                metrics_row["accuracy"],
                "accuracy",
                option_id
            ),

            "latency_ms": to_nonnegative_float(
                metrics_row["latency_ms"],
                "latency_ms",
                option_id
            ),

            "measured_energy_kwh": to_nonnegative_float(
                metrics_row["measured_energy_kwh"],
                "measured_energy_kwh",
                option_id
            ),

            "training_kgco2e": to_nonnegative_float(
                carbon_row["training_kgco2e"],
                "training_kgco2e",
                option_id
            ),

            "inference_kgco2e": to_nonnegative_float(
                carbon_row["inference_kgco2e"],
                "inference_kgco2e",
                option_id
            ),

            "storage_kgco2e": to_nonnegative_float(
                carbon_row["storage_kgco2e"],
                "storage_kgco2e",
                option_id
            ),

            "networking_kgco2e": to_nonnegative_float(
                carbon_row["networking_kgco2e"],
                "networking_kgco2e",
                option_id
            ),

            "retraining_kgco2e": to_nonnegative_float(
                carbon_row["retraining_kgco2e"],
                "retraining_kgco2e",
                option_id
            ),

            "hardware_kgco2e": to_nonnegative_float(
                carbon_row["hardware_kgco2e"],
                "hardware_kgco2e",
                option_id
            ),

            "total_kgco2e": to_nonnegative_float(
                carbon_row["total_kgco2e"],
                "total_kgco2e",
                option_id
            ),

            "low_kgco2e": to_nonnegative_float(
                carbon_row["low_kgco2e"],
                "low_kgco2e",
                option_id
            ),

            "high_kgco2e": to_nonnegative_float(
                carbon_row["high_kgco2e"],
                "high_kgco2e",
                option_id
            )
        }

        if option["accuracy"] > 1:
            raise ValueError(
                f"Accuracy for {option_id} must be between 0 and 1."
            )

        merged_options.append(option)

    return merged_options