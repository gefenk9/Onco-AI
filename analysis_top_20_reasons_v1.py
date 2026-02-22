#!/usr/bin/env python3
"""Analysis Top 20 Treatment Reasons v1 - Classify patients into 3 treatment types and identify top 20 reasons for treatment decisions."""

import argparse
import csv
import sys


# Required columns for input CSV
REQUIRED_COLUMNS = ["PatId", "Current_Disease", "Summary_Conclusions", "Recommendations"]
INPUT_CSV_PATH = "./cases.csv"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify patients into treatment types and identify top 20 reasons for treatment decisions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="azure_openai",
        choices=["azure_openai", "bedrock", "anthropic"],
        help="LLM provider to use for analysis",
    )
    return parser.parse_args()


def read_and_validate_csv(file_path: str) -> list[dict[str, str]]:
    """Read CSV file and validate its structure."""
    try:
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            # Check if reader has fieldnames
            if not reader.fieldnames:
                print(f"ERROR: Could not read column headers from '{file_path}'")
                sys.exit(1)

            # Check if required columns exist
            missing_columns = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
            if missing_columns:
                print(
                    f"ERROR: Required columns not found in CSV: {missing_columns}. "
                    f"Available columns: {reader.fieldnames}"
                )
                sys.exit(1)

            # Print warning if column names don't match exactly
            if set(reader.fieldnames) != set(REQUIRED_COLUMNS):
                print(
                    f"WARNING: CSV headers in '{file_path}' are {reader.fieldnames}, "
                    f"expected {REQUIRED_COLUMNS}"
                )

            # Read all rows
            patients = list(reader)

        return patients

    except FileNotFoundError:
        print(f"ERROR: Input CSV file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read CSV file '{file_path}': {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()

    # Print selected provider
    print(f"LLM Provider: {args.provider}")
    print("--- Analysis Top 20 Treatment Reasons v1 ---")

    # Read and validate CSV
    patients = read_and_validate_csv(INPUT_CSV_PATH)

    # Print total patient count
    print(f"Total patients to process: {len(patients)}")


if __name__ == "__main__":
    main()
