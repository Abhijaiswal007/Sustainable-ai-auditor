import csv
from pathlib import Path


SCENARIOS = ("low", "base", "high")


def validate_nonnegative(**values):
    """Reject negative numeric inputs."""

    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")


def read_indexed_csv(file_path, key_column):
    """
    Read a CSV and return a dictionary indexed by one column.

    Example:
        rows["India"]
        rows["NVIDIA T4"]
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = {}

        for row in reader:
            key = row[key_column].strip()
            rows[key] = row

    return rows


def load_reference_data(data_directory):
    """Load all reference CSV files."""

    data_directory = Path(data_directory)

    return {
        "grid": read_indexed_csv(
            data_directory / "grid_intensity.csv",
            "region"
        ),
        "hardware": read_indexed_csv(
            data_directory / "hardware.csv",
            "device"
        ),
        "assumptions": read_indexed_csv(
            data_directory / "assumptions.csv",
            "parameter"
        ),
    }


def carbon_from_energy(energy_kwh, grid_intensity_g_per_kwh):
    """Convert electricity consumption into kg CO2e."""

    validate_nonnegative(
        energy_kwh=energy_kwh,
        grid_intensity=grid_intensity_g_per_kwh
    )

    return energy_kwh * grid_intensity_g_per_kwh / 1000


def training_carbon(
    power_w,
    hours,
    grid_intensity_g_per_kwh,
    gpu_count=1,
    utilization=0.8,
    pue=1.2
):
    """Calculate training energy and carbon."""

    validate_nonnegative(
        power_w=power_w,
        hours=hours,
        grid_intensity=grid_intensity_g_per_kwh,
        gpu_count=gpu_count,
        utilization=utilization,
        pue=pue
    )

    energy_kwh = (
        power_w
        * hours
        * gpu_count
        * utilization
        * pue
    ) / 1000

    carbon_kg = carbon_from_energy(
        energy_kwh,
        grid_intensity_g_per_kwh
    )

    return {
        "energy_kwh": energy_kwh,
        "carbon_kgco2e": carbon_kg
    }


def inference_carbon(
    energy_per_request_kwh,
    requests_per_day,
    days,
    grid_intensity_g_per_kwh,
    pue=1.2
):
    """Calculate inference energy and carbon."""

    validate_nonnegative(
        energy_per_request_kwh=energy_per_request_kwh,
        requests_per_day=requests_per_day,
        days=days,
        grid_intensity=grid_intensity_g_per_kwh,
        pue=pue
    )

    number_of_requests = requests_per_day * days

    energy_kwh = (
        energy_per_request_kwh
        * number_of_requests
        * pue
    )

    carbon_kg = carbon_from_energy(
        energy_kwh,
        grid_intensity_g_per_kwh
    )

    return {
        "energy_kwh": energy_kwh,
        "carbon_kgco2e": carbon_kg
    }


def storage_carbon(
    storage_gb,
    months,
    energy_kwh_per_gb_month,
    grid_intensity_g_per_kwh,
    pue=1.2
):
    """Calculate storage energy and carbon."""

    validate_nonnegative(
        storage_gb=storage_gb,
        months=months,
        energy_factor=energy_kwh_per_gb_month,
        grid_intensity=grid_intensity_g_per_kwh,
        pue=pue
    )

    energy_kwh = (
        storage_gb
        * months
        * energy_kwh_per_gb_month
        * pue
    )

    carbon_kg = carbon_from_energy(
        energy_kwh,
        grid_intensity_g_per_kwh
    )

    return {
        "energy_kwh": energy_kwh,
        "carbon_kgco2e": carbon_kg
    }


def networking_carbon(
    transferred_gb,
    energy_kwh_per_gb,
    grid_intensity_g_per_kwh
):
    """Calculate network data-transfer energy and carbon."""

    validate_nonnegative(
        transferred_gb=transferred_gb,
        energy_factor=energy_kwh_per_gb,
        grid_intensity=grid_intensity_g_per_kwh
    )

    energy_kwh = transferred_gb * energy_kwh_per_gb

    carbon_kg = carbon_from_energy(
        energy_kwh,
        grid_intensity_g_per_kwh
    )

    return {
        "energy_kwh": energy_kwh,
        "carbon_kgco2e": carbon_kg
    }


def hardware_carbon(
    embodied_carbon_kg,
    project_device_hours,
    lifetime_years,
    device_count=1,
    lifetime_utilization=0.7
):
    """Allocate part of the hardware manufacturing footprint."""

    validate_nonnegative(
        embodied_carbon_kg=embodied_carbon_kg,
        project_device_hours=project_device_hours,
        lifetime_years=lifetime_years,
        device_count=device_count,
        lifetime_utilization=lifetime_utilization
    )

    if lifetime_years == 0:
        raise ValueError("Hardware lifetime must be greater than zero.")

    if lifetime_utilization == 0:
        raise ValueError(
            "Hardware lifetime utilization must be greater than zero."
        )

    lifetime_device_hours = (
        lifetime_years
        * 365
        * 24
        * lifetime_utilization
    )

    allocated_carbon_kg = (
        embodied_carbon_kg
        * device_count
        * project_device_hours
        / lifetime_device_hours
    )

    return {
        "energy_kwh": 0.0,
        "carbon_kgco2e": allocated_carbon_kg
    }


def calculate_lifecycle(system, reference_data, scenario="base"):
    """Calculate one complete lifecycle scenario."""

    if scenario not in SCENARIOS:
        raise ValueError(
            f"Scenario must be one of: {', '.join(SCENARIOS)}"
        )

    region = system["region"]
    gpu_name = system["gpu_name"]

    if region not in reference_data["grid"]:
        raise ValueError(f"Unknown region: {region}")

    if gpu_name not in reference_data["hardware"]:
        raise ValueError(f"Unknown hardware device: {gpu_name}")

    grid_row = reference_data["grid"][region]
    hardware_row = reference_data["hardware"][gpu_name]
    assumptions = reference_data["assumptions"]

    grid_intensity = float(
        grid_row[f"{scenario}_gco2_per_kwh"]
    )

    pue = float(assumptions["pue"][scenario])

    storage_energy_factor = float(
        assumptions["storage_energy"][scenario]
    )

    network_energy_factor = float(
        assumptions["network_energy"][scenario]
    )

    gpu_utilization = float(
        assumptions["gpu_utilization"][scenario]
    )

    lifetime_utilization = float(
        assumptions["hardware_lifetime_utilization"][scenario]
    )

    power_w = float(hardware_row["power_w"])

    embodied_carbon_kg = float(
        hardware_row[f"embodied_{scenario}_kg"]
    )

    lifetime_years = float(
        hardware_row["lifetime_years"]
    )

    gpu_count = system.get("gpu_count", 1)
    days = system.get("days", 365)
    months = system.get("months", 12)

    initial_training_hours = (
        system["training_hours"]
        * system.get("training_runs", 1)
    )

    retraining_hours = (
        system.get("retraining_hours_per_run", 0)
        * system.get("retraining_runs_per_year", 0)
    )

    training_result = training_carbon(
        power_w=power_w,
        hours=initial_training_hours,
        grid_intensity_g_per_kwh=grid_intensity,
        gpu_count=gpu_count,
        utilization=gpu_utilization,
        pue=pue
    )

    retraining_result = training_carbon(
        power_w=power_w,
        hours=retraining_hours,
        grid_intensity_g_per_kwh=grid_intensity,
        gpu_count=gpu_count,
        utilization=gpu_utilization,
        pue=pue
    )

    inference_result = inference_carbon(
        energy_per_request_kwh=system["energy_per_request_kwh"],
        requests_per_day=system["requests_per_day"],
        days=days,
        grid_intensity_g_per_kwh=grid_intensity,
        pue=pue
    )

    storage_result = storage_carbon(
        storage_gb=system["storage_gb"],
        months=months,
        energy_kwh_per_gb_month=storage_energy_factor,
        grid_intensity_g_per_kwh=grid_intensity,
        pue=pue
    )

    total_network_gb = (
        system["network_gb_per_month"]
        * months
    )

    networking_result = networking_carbon(
        transferred_gb=total_network_gb,
        energy_kwh_per_gb=network_energy_factor,
        grid_intensity_g_per_kwh=grid_intensity
    )

    inference_device_hours = (
        system.get("inference_device_hours_per_day", 0)
        * days
    )

    total_project_device_hours = (
        initial_training_hours
        + retraining_hours
        + inference_device_hours
    )

    hardware_result = hardware_carbon(
        embodied_carbon_kg=embodied_carbon_kg,
        project_device_hours=total_project_device_hours,
        lifetime_years=lifetime_years,
        device_count=gpu_count,
        lifetime_utilization=lifetime_utilization
    )

    stages = {
        "training": training_result,
        "inference": inference_result,
        "storage": storage_result,
        "networking": networking_result,
        "retraining": retraining_result,
        "hardware": hardware_result
    }

    total_energy_kwh = sum(
        stage["energy_kwh"]
        for stage in stages.values()
    )

    total_carbon_kg = sum(
        stage["carbon_kgco2e"]
        for stage in stages.values()
    )

    return {
        "scenario": scenario,
        "region": region,
        "gpu_name": gpu_name,
        "grid_intensity_g_per_kwh": grid_intensity,
        "stages": stages,
        "total_energy_kwh": total_energy_kwh,
        "total_kgco2e": total_carbon_kg
    }


def calculate_all_scenarios(system, reference_data):
    """Calculate low, expected and high results."""

    results = {
        scenario: calculate_lifecycle(
            system=system,
            reference_data=reference_data,
            scenario=scenario
        )
        for scenario in SCENARIOS
    }

    return {
        "low": results["low"],
        "expected": results["base"],
        "high": results["high"]
    }