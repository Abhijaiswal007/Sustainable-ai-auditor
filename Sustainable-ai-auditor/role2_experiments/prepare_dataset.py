"""
Role 2 - Dataset preparation
Builds a fixed, reproducible SST-2 validation subset for fair model comparison.
Do not rerun this after results exist without explicit approval - it will
invalidate any prior benchmark comparisons.
"""

import os
import sys
import pandas as pd
from datasets import load_dataset

SEED = 42
SAMPLE_COUNT = 500
OUTPUT_PATH = os.path.join("data", "test_data.csv")


def main():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        print(f"'{OUTPUT_PATH}' already exists.")
        answer = input(
            "Overwriting will invalidate any existing benchmark results "
            "that used this file. Type 'yes' to overwrite, anything else to abort: "
        )
        if answer.strip().lower() != "yes":
            print("Aborted. Existing dataset was not modified.")
            sys.exit(1)

    print("Downloading SST-2 validation split...")
    dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")

    if SAMPLE_COUNT > len(dataset):
        print(
            f"Requested {SAMPLE_COUNT} samples but validation split only "
            f"has {len(dataset)}. Reduce SAMPLE_COUNT."
        )
        sys.exit(1)

    # Fixed, reproducible subset
    shuffled = dataset.shuffle(seed=SEED)
    subset = shuffled.select(range(SAMPLE_COUNT))

    df = pd.DataFrame({
        "text": subset["sentence"],
        "label": subset["label"],
    })

    # Validation checks before saving
    if df["text"].isnull().any() or df["label"].isnull().any():
        print("Missing text or label values found. Aborting save.")
        sys.exit(1)

    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        print(f"Unexpected label values found: {unique_labels}. Expected only 0 and 1.")
        sys.exit(1)

    df.to_csv(OUTPUT_PATH, index=False)

    positive_count = int((df["label"] == 1).sum())
    negative_count = int((df["label"] == 0).sum())

    print(f"Saved {len(df)} rows to '{OUTPUT_PATH}'")
    print(f"Positive (label=1): {positive_count}")
    print(f"Negative (label=0): {negative_count}")
    print(f"Seed used: {SEED}")


if __name__ == "__main__":
    main()