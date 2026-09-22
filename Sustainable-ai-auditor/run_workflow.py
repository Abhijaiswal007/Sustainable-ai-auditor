import subprocess
import sys
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parent

CARBON_SCRIPT = (
    PROJECT_DIRECTORY
    / "carbon_engine"
    / "generate_model_carbon.py"
)

DECISION_SCRIPT = (
    PROJECT_DIRECTORY
    / "decision_engine"
    / "main.py"
)


def run_script(script_path):
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent,
        check=True,
    )


def main():
    print("STEP 1: CALCULATING MODEL LIFECYCLE CARBON")
    print("=" * 70)
    run_script(CARBON_SCRIPT)

    print("\nSTEP 2: RUNNING THE SUSTAINABLE AI DECISION ENGINE")
    print("=" * 70)
    run_script(DECISION_SCRIPT)


if __name__ == "__main__":
    main()
