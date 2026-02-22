#!/usr/bin/env python3
"""Analysis Top 20 Treatment Reasons v1 - Classify patients into 3 treatment types and identify top 20 reasons for treatment decisions."""

import argparse
import csv
import sys

from llm_client import invoke_llm


# Required columns for input CSV
REQUIRED_COLUMNS = ["PatId", "Current_Disease", "Summary_Conclusions", "Recommendations"]
INPUT_CSV_PATH = "./cases.csv"

# Hebrew system prompt for patient classification
SYSTEM_PROMPT = (
    "אתה רופא אונקולוג מומחה. עליך לבחון תיק מטופל ולקבוע את סוג הטיפול שהמטופל קיבל "
    "על סמך NCCN וESMO גידלינים. "
    "עליך להבחין בין שלושה סוגי טיפול בלבד: "
    "1. כימותרפיה ואימונותרפיה "
    "2. אימונותרפיה בלבד "
    "3. אימונותרפיה וכימותרפיה במינון מופחת "
    "בנוסף, עליך לציין את הסיבה העיקרית אחת בלבד שהובילה להחלטת הטיפול. "
    "ענה בעברית בלבד."
    "פורמט התשובה:"
    "סוג טיפול: [אחד משלושת הסוגים]"
    "סיבה עיקרית: [סיבה אחת]"
)


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


def classify_patient(
    pat_id: str,
    current_disease: str,
    summary_conclusions: str,
    recommendations: str,
    provider: str,
) -> str:
    """
    Classify a patient into treatment type using LLM.

    Returns the LLM response string for later processing.
    """
    user_prompt = (
        f"כך סיכם הרופא את המקרה:\n"
        f"מחלה נוכחית: {current_disease}\n\n"
        f"סיכומים ומסקנות: {summary_conclusions}\n\n"
        f"המלצות: {recommendations}"
    )

    response = invoke_llm(
        system_prompt=SYSTEM_PROMPT,
        user_prompt_text=user_prompt,
        max_tokens=1000,
        temperature=0.0,
        provider_override=provider,
    )

    return response


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

    # Process each patient (for now, just make LLM calls)
    for i, patient in enumerate(patients, 1):
        pat_id = patient.get("PatId", "")
        current_disease = patient.get("Current_Disease", "")
        summary_conclusions = patient.get("Summary_Conclusions", "")
        recommendations = patient.get("Recommendations", "")

        print(f"\n--- Processing patient {i}/{len(patients)} (PatId: {pat_id}) ---")

        # Make first LLM call for patient classification
        llm_response = classify_patient(
            pat_id=pat_id,
            current_disease=current_disease,
            summary_conclusions=summary_conclusions,
            recommendations=recommendations,
            provider=args.provider,
        )

        print(f"LLM Response: {llm_response}")


if __name__ == "__main__":
    main()
