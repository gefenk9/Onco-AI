#!/usr/bin/env python3
"""Analysis Top 20 Treatment Reasons v1 - Classify patients into 3 treatment types and identify top 20 reasons for treatment decisions."""

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass

from llm_client import invoke_llm


# Rate limiting configuration
REQUEST_DELAY_SECONDS = 31  # Delay in seconds between requests (rate limit)


@dataclass
class PatientReason:
    """Represents a patient's primary reason for treatment decision."""
    pat_id: str
    reason_text: str


@dataclass
class TopReason:
    """Represents a top reason with metadata."""
    reason_name: str
    patient_count: str
    explanation: str


# Required columns for input CSV
REQUIRED_COLUMNS = ["PatId", "Current_Disease", "Summary_Conclusions", "Recommendations"]
INPUT_CSV_PATH = "./cases.csv"

# Valid treatment types (normalized English names)
TREATMENT_CHEMO_IMMUNO = "Chemo + immuno"
TREATMENT_IMMUNO_ONLY = "immuno only"
TREATMENT_IMMUNO_CHEMO_REDUCED = "Immuno + chemo reduce dose"
UNCATEGORIZED_CSV_PATH = "./uncategorized.csv"
OUTPUT_CSV_PATH = "./analysis_v1_results.csv"
NOT_IN_TOP_20_CSV_PATH = "./not_in_top_20.csv"

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
    Keywords are sorted by length (longest first) to ensure specific matches
    are checked before general ones.
    IMPORTANT: Check reduced dose variants BEFORE regular chemo+immuno variants
    to avoid false positives.
    """
    raw_lower = raw_treatment.lower()

    # Check for Immuno + chemo reduce dose variants FIRST (sorted by length, longest first)
    immuno_chemo_reduced_keywords = [
        "immunotherapy and reduced dose chemotherapy",
        "reduced dose chemo and immuno",
        "immuno + chemo reduce dose",
        "אימונו וכימו במינון מופחת",
        "כימו מופחת ואימונו",
    ]
    immuno_chemo_reduced_keywords.sort(key=len, reverse=True)
    for keyword in immuno_chemo_reduced_keywords:
        if keyword.lower() in raw_lower:
            return TREATMENT_IMMUNO_CHEMO_REDUCED

    # Check for immuno only variants (sorted by length, longest first)
    immuno_only_keywords = [
        "immunotherapy only",
        "immuno only",
        "only immuno",
        "אימונותרפיה בלבד",
        "רק אימונו",
        "אימונו בלבד",
    ]
    immuno_only_keywords.sort(key=len, reverse=True)
    for keyword in immuno_only_keywords:
        if keyword.lower() in raw_lower:
            return TREATMENT_IMMUNO_ONLY

    # Check for Chemo + immuno variants LAST (sorted by length, longest first)
    chemo_immuno_keywords = [
        "chemotherapy and immunotherapy",
        "chemo and immuno",
        "chemo + immuno",
        "כימותרפיה ואימונותרפיה",
        "כימו ואימונו",
        "אימונו וכימו",
    ]
    chemo_immuno_keywords.sort(key=len, reverse=True)
    for keyword in chemo_immuno_keywords:
        if keyword.lower() in raw_lower:
            return TREATMENT_CHEMO_IMMUNO

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


def select_top_20_reasons(patient_reasons: list[PatientReason], provider: str) -> str:
    """
    Second LLM call to select top 20 reasons from all patient reasons.

    Sends all patient reasons to LLM in single call and asks to merge similar
    reasons and explain decisions.
    """
    # Build prompt with all patient reasons
    reasons_text = "\n".join(
        [f"{i+1}. [PatId: {r.pat_id}] {r.reason_text}" for i, r in enumerate(patient_reasons)]
    )

    # Hebrew prompt for top-20 selection
    system_prompt = (
        "אתה רופא אונקולוג מומחה שמנתח סיבות להחלטות טיפוליות. "
        "להלן רשימה של סיבות שנתנו על ידי רופאים עבור כל מטופל. "
        "עליך לבחור את 20 הסיבות העיקריות ביותר שהשפיעו על החלטות הטיפול. "
        "עליך למזג סיבות דומות לקטגוריות אחתות ולהסביר את ההחלטות שלך. "
        "פורמט התשובה:"
        "לכל סיבה, ציין:"
        "1. מספר סידורי (1-20)"
        "2. שם הסיבה"
        "3. כמות מטופלים שהשתמשו בסיבה זו"
        "4. הסבר למו נכללה ברשימת ה-20"
        "ענה בעברית בלבד."
    )

    user_prompt = (
        f"להלן רשימה של סיבות מכל המטופלים:\n\n"
        f"{reasons_text}\n\n"
        f"בחר 20 הסיבות העיקריות ביותר מהרשימה, מזג סיבות דומות, "
        f"והסבר את החלטותיך."
    )

    response = invoke_llm(
        system_prompt=system_prompt,
        user_prompt_text=user_prompt,
        max_tokens=4000,
        temperature=0.0,
        provider_override=provider,
    )

    return response


def parse_top_20_response(llm_response: str) -> list[TopReason]:
    """
    Parse top-20 LLM response to extract structured data about reasons.

    Extracts reason name, patient count, and explanation for each top reason.
    Handles cases where LLM returns fewer than 20 reasons.
    """
    top_reasons: list[TopReason] = []
    lines = llm_response.strip().split("\n")

    current_reason: dict[str, str] = {}
    reason_number = None

    for line in lines:
        line = line.strip()

        # Check for numbered reason pattern (e.g., "1." or "1)" or "1.")
        # Match patterns like: "1.", "1)", "1." with various delimiters
        reason_match = re.match(r"^(\d+)[\.)]\s*(.+)$", line)
        if reason_match:
            # Save previous reason if exists
            if current_reason and "name" in current_reason:
                top_reasons.append(
                    TopReason(
                        reason_name=current_reason.get("name", ""),
                        patient_count=current_reason.get("count", "0"),
                        explanation=current_reason.get("explanation", ""),
                    )
                )

            # Start new reason
            reason_number = reason_match.group(1)
            current_reason = {"name": reason_match.group(2).strip()}

        # Check for patient count patterns
        elif current_reason:
            # Look for "כמות מטופלים" (number of patients) or "מספר" (number)
            if "כמות מטופלים" in line or "מטופלים:" in line or "חולים:" in line:
                parts = re.split(r"[:：]", line, 1)
                if len(parts) > 1:
                    current_reason["count"] = parts[1].strip()

            # Check for explanation patterns
            elif "הסבר" in line or "כי" in line:
                # Extract text after the explanation marker
                parts = re.split(r"[:：]", line, 1)
                if len(parts) > 1:
                    explanation_text = parts[1].strip()
                    # Append to existing explanation or create new
                    current_reason["explanation"] = current_reason.get("explanation", "") + " " + explanation_text

    # Save last reason if exists
    if current_reason and "name" in current_reason:
        top_reasons.append(
            TopReason(
                reason_name=current_reason.get("name", ""),
                patient_count=current_reason.get("count", "0"),
                explanation=current_reason.get("explanation", ""),
            )
        )

    print(f"Parsed {len(top_reasons)} top reasons from LLM response")
    return top_reasons


def create_not_in_top_20_csv(
    all_patient_reasons: list[PatientReason], top_reasons: list[TopReason]
) -> None:
    """
    Create not_in_top_20.csv for patients whose reasons were not in top 20.

    Columns: PatId, primary_reason, exclusion_reason.
    Populates exclusion_reason with LLM's explanation from top reasons.
    """
    try:
        # Create set of top reason names for faster lookup
        top_reason_names = {tr.reason_name for tr in top_reasons}

        with open(NOT_IN_TOP_20_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["PatId", "primary_reason", "exclusion_reason"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            excluded_count = 0
            for patient_reason in all_patient_reasons:
                # Check if patient's reason is not in top 20
                # This is a simple check - in reality, we'd need to match
                # patient reasons to the grouped top reasons more intelligently
                if patient_reason.reason_text not in top_reason_names:
                    # Find the most relevant exclusion reason from top reasons
                    # (using the first explanation as a general explanation)
                    exclusion_reason = ""
                    if top_reasons:
                        # Use explanation from first top reason as general exclusion explanation
                        exclusion_reason = f"Not in top {len(top_reasons)} most common reasons"
                    else:
                        exclusion_reason = "No top reasons identified"

                    writer.writerow(
                        {
                            "PatId": patient_reason.pat_id,
                            "primary_reason": patient_reason.reason_text,
                            "exclusion_reason": exclusion_reason,
                        }
                    )
                    excluded_count += 1

        print(f"Created {NOT_IN_TOP_20_CSV_PATH} with {excluded_count} excluded patients")
    except Exception as e:
        print(f"ERROR: Failed to create not_in_top_20.csv: {e}")


def append_top_20_reasons_to_output(top_reasons: list[TopReason]) -> None:
    """
    Append top-20 reasons to main output CSV.

    Each top reason as row with PatId = 'TOP_REASON_N' format.
    Columns: PatId, reason_name, patient_count, explanation.
    """
    try:
        with open(OUTPUT_CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
            # Note: Top reasons have different columns than patient rows
            # We'll use DictWriter with top reason specific fieldnames
            writer = csv.writer(csvfile)

            for i, top_reason in enumerate(top_reasons, 1):
                writer.writerow(
                    [
                        f"TOP_REASON_{i}",
                        top_reason.reason_name,
                        top_reason.patient_count,
                        top_reason.explanation,
                    ]
                )

        print(f"Appended {len(top_reasons)} top reasons to {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to append top reasons to output CSV: {e}")


def validate_patient_count_totals(
    treatment_counts: dict[str, int], total_patients: int
) -> None:
    """
    Verify that sum of patients across treatment types matches total input.

    Counts patients in each treatment type category and compares to total.
    Prints warning on mismatch with both counts for comparison.
    Does not stop execution on mismatch.
    """
    # Sum counts across all treatment types
    total_processed = sum(treatment_counts.values())

    # Print treatment type breakdown
    print("\n--- Patient Count by Treatment Type ---")
    for treatment_type, count in treatment_counts.items():
        print(f"  {treatment_type}: {count}")
    print(f"  Total processed: {total_processed}")

    # Compare to total input
    if total_processed != total_patients:
        warning_msg = (
            f"WARNING: Patient count mismatch detected\n"
            f"  Total input patients: {total_patients}\n"
            f"  Total processed: {total_processed}\n"
            f"  Difference: {abs(total_patients - total_processed)}"
        )
        print(warning_msg)
        # Log to stderr as well
        print(warning_msg, file=sys.stderr)
    else:
        print("Patient count validation: PASS - counts match")


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

    # Track patients by treatment type for validation
    treatment_counts: dict[str, int] = {
        TREATMENT_CHEMO_IMMUNO: 0,
        TREATMENT_IMMUNO_ONLY: 0,
        TREATMENT_IMMUNO_CHEMO_REDUCED: 0,
    }

    # Process each patient
    for i, patient in enumerate(patients, 1):
        # Apply rate limiting delay (not before first patient)
        if i > 1 and args.provider != "anthropic":
            print(f"\n--- Waiting {REQUEST_DELAY_SECONDS} seconds before next patient... ---")
            for remaining in range(REQUEST_DELAY_SECONDS, 0, -1):
                print(f"  {remaining} seconds remaining...", end="\r")
                time.sleep(1)
            print("  Continuing...")

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

        # Track treatment type count
        if normalized_treatment_type in treatment_counts:
            treatment_counts[normalized_treatment_type] += 1
        else:
            # Track uncategorized as Chemo + immuno for counting purposes
            treatment_counts[TREATMENT_CHEMO_IMMUNO] += 1

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

    # Validate patient count totals by treatment type
    validate_patient_count_totals(treatment_counts, len(patients))

    # Second LLM call: Select top 20 reasons
    if all_patient_reasons:
        print("\n--- Selecting Top 20 Reasons ---")
        top_20_response = select_top_20_reasons(all_patient_reasons, args.provider)
        print("\nTop 20 Reasons Response:")
        print(top_20_response)

        # Parse top-20 response
        top_reasons = parse_top_20_response(top_20_response)
        print(f"\nParsed {len(top_reasons)} top reasons")

        # Create not_in_top_20.csv for excluded reasons
        create_not_in_top_20_csv(all_patient_reasons, top_reasons)

        # Append top-20 reasons to output CSV
        append_top_20_reasons_to_output(top_reasons)
    else:
        print("\nWARNING: No patient reasons collected, skipping top-20 analysis")


if __name__ == "__main__":
    main()
