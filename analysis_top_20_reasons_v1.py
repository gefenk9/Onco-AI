#!/usr/bin/env python3
"""Analysis Top 20 Treatment Reasons v1 - Classify patients into 3 treatment types and identify top 20 reasons for treatment decisions."""

import argparse
import csv
import re
import sys
from dataclasses import dataclass

from llm_client import invoke_llm


@dataclass
class PatientReason:
    """Represents a patient's primary reason for treatment decision."""
    pat_id: str
    reason_text: str


# Required columns for input CSV
REQUIRED_COLUMNS = ["PatId", "Current_Disease", "Summary_Conclusions", "Recommendations"]
INPUT_CSV_PATH = "./cases.csv"

# Valid treatment types (normalized English names)
TREATMENT_CHEMO_IMMUNO = "Chemo + immuno"
TREATMENT_IMMUNO_ONLY = "immuno only"
TREATMENT_IMMUNO_CHEMO_REDUCED = "Immuno + chemo reduce dose"
UNCATEGORIZED_CSV_PATH = "./uncategorized.csv"
OUTPUT_CSV_PATH = "./analysis_v1_results.csv"

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


def is_valid_treatment_type(treatment_type: str) -> bool:
    """Check if treatment type is one of the valid normalized types."""
    return treatment_type in [TREATMENT_CHEMO_IMMUNO, TREATMENT_IMMUNO_ONLY, TREATMENT_IMMUNO_CHEMO_REDUCED]


def write_uncategorized_patient(pat_id: str, original_response: str, normalized_as: str) -> None:
    """Write uncategorized patient to uncategorized.csv."""
    try:
        # Check if file exists to determine if we need to write headers
        file_exists = False
        try:
            with open(UNCATEGORIZED_CSV_PATH, "r", encoding="utf-8"):
                file_exists = True
        except FileNotFoundError:
            pass

        with open(UNCATEGORIZED_CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["PatId", "original_response", "normalized_as"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {"PatId": pat_id, "original_response": original_response, "normalized_as": normalized_as}
            )
    except Exception as e:
        print(f"  ERROR: Failed to write to uncategorized.csv: {e}")


def initialize_output_csv() -> None:
    """Create output CSV file with headers if it doesn't exist."""
    try:
        # Check if file exists
        try:
            with open(OUTPUT_CSV_PATH, "r", encoding="utf-8"):
                # File exists, don't overwrite
                return
        except FileNotFoundError:
            pass

        # Create file with headers
        with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["PatId", "treatment_type", "primary_reason"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        print(f"Created output file: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to initialize output CSV: {e}")


def save_patient_result(pat_id: str, treatment_type: str, primary_reason: str) -> None:
    """Save patient result to output CSV file incrementally."""
    try:
        with open(OUTPUT_CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["PatId", "treatment_type", "primary_reason"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(
                {"PatId": pat_id, "treatment_type": treatment_type, "primary_reason": primary_reason}
            )
    except Exception as e:
        print(f"  ERROR: Failed to write patient result: {e}")


def normalize_treatment_type(raw_treatment: str) -> str:
    """
    Normalize treatment type to standard English strings.

    Handles flexible Hebrew/English input with simple keyword matching.
    """
    raw_lower = raw_treatment.lower()

    # Check for Chemo + immuno variants
    chemo_immuno_keywords = [
        "כימותרפיה ואימונותרפיה",
        "כימו ואימונו",
        "אימונו וכימו",
        "chemo + immuno",
        "chemo and immuno",
        "chemotherapy and immunotherapy",
    ]
    for keyword in chemo_immuno_keywords:
        if keyword.lower() in raw_lower:
            return TREATMENT_CHEMO_IMMUNO

    # Check for immuno only variants
    immuno_only_keywords = [
        "אימונותרפיה בלבד",
        "רק אימונו",
        "אימונו בלבד",
        "immuno only",
        "immunotherapy only",
        "only immuno",
    ]
    for keyword in immuno_only_keywords:
        if keyword.lower() in raw_lower:
            return TREATMENT_IMMUNO_ONLY

    # Check for Immuno + chemo reduce dose variants
    immuno_chemo_reduced_keywords = [
        "אימונו וכימו במינון מופחת",
        "כימו מופחת ואימונו",
        "immuno + chemo reduce dose",
        "immunotherapy and reduced dose chemotherapy",
        "reduced dose chemo and immuno",
    ]
    for keyword in immuno_chemo_reduced_keywords:
        if keyword.lower() in raw_lower:
            return TREATMENT_IMMUNO_CHEMO_REDUCED

    # Return raw value if no match
    return raw_treatment


def parse_llm_response(pat_id: str, llm_response: str) -> tuple[str, str, str]:
    """
    Parse LLM response to extract treatment type and primary reason.

    Returns tuple of (raw_treatment_type, normalized_treatment_type, primary_reason).
    Handles invalid treatment types by normalizing to 'Chemo + immuno' and logging to uncategorized.csv.
    """
    lines = llm_response.strip().split("\n")

    raw_treatment_type = ""
    primary_reason = ""

    for line in lines:
        line = line.strip()

        # Extract treatment type
        if line.startswith("סוג טיפול:") or line.startswith("סוג טיפול :"):
            raw_treatment_type = line.replace("סוג טיפול:", "").replace("סוג טיפול :", "").strip()
        elif line.startswith("טיפול:") or line.startswith("טיפול :"):
            raw_treatment_type = line.replace("טיפול:", "").replace("טיפול :", "").strip()
        elif "treatment type:" in line.lower():
            # Handle English variants
            parts = line.split(":", 1)
            if len(parts) > 1:
                raw_treatment_type = parts[1].strip()

        # Extract primary reason
        elif line.startswith("סיבה עיקרית:") or line.startswith("סיבה עיקרית :"):
            primary_reason = line.replace("סיבה עיקרית:", "").replace("סיבה עיקרית :", "").strip()
        elif line.startswith("סיבה:") or line.startswith("סיבה :"):
            primary_reason = line.replace("סיבה:", "").replace("סיבה :", "").strip()
        elif "primary reason:" in line.lower() or "reason:" in line.lower():
            # Handle English variants
            parts = line.split(":", 1)
            if len(parts) > 1:
                primary_reason = parts[1].strip()

    # Normalize treatment type
    normalized_treatment_type = normalize_treatment_type(raw_treatment_type)

    # Check if treatment type is valid, otherwise classify as uncategorized
    if not is_valid_treatment_type(normalized_treatment_type):
        print(f"  WARNING: Treatment type '{raw_treatment_type}' could not be normalized to a valid category")
        print(f"  Classifying as '{TREATMENT_CHEMO_IMMUNO}' (default)")
        # Write to uncategorized.csv
        write_uncategorized_patient(pat_id, llm_response, TREATMENT_CHEMO_IMMUNO)
        # Use default treatment type
        normalized_treatment_type = TREATMENT_CHEMO_IMMUNO
    elif raw_treatment_type != normalized_treatment_type:
        print(f"  Normalized treatment type: '{raw_treatment_type}' -> '{normalized_treatment_type}'")
    else:
        print(f"  Treatment type: {normalized_treatment_type}")

    print(f"  Primary reason: {primary_reason}")

    return raw_treatment_type, normalized_treatment_type, primary_reason


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

    # Initialize output CSV file with headers
    initialize_output_csv()

    # Collect all patient reasons for top-20 analysis
    all_patient_reasons: list[PatientReason] = []

    # Process each patient
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

        # Parse and normalize the LLM response
        raw_treatment_type, normalized_treatment_type, primary_reason = parse_llm_response(
            pat_id, llm_response
        )

        # Save patient result immediately after processing
        save_patient_result(pat_id, normalized_treatment_type, primary_reason)

        # Collect patient reason for top-20 analysis
        all_patient_reasons.append(PatientReason(pat_id=pat_id, reason_text=primary_reason))

    # Print count of total reasons collected
    print(f"\n--- Summary ---")
    print(f"Total patients processed: {len(patients)}")
    print(f"Total reasons collected: {len(all_patient_reasons)}")

    # Verify count matches
    if len(all_patient_reasons) != len(patients):
        print(
            f"WARNING: Mismatch between patients processed ({len(patients)}) "
            f"and reasons collected ({len(all_patient_reasons)})"
        )
    else:
        print("Reasons count matches patients processed")


if __name__ == "__main__":
    main()
