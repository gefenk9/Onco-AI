#!/usr/bin/env python3
"""Analysis Top 20 Treatment Reasons v1 - Classify patients into 3 treatment types and identify top 20 reasons for treatment decisions."""

import argparse
import csv
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from llm_client import invoke_llm


# Rate limiting configuration
REQUEST_DELAY_SECONDS = 31  # Delay in seconds between requests (rate limit)
MAX_REASON_WORDS = 3
MAX_REASON_TRIES = 3
REASON_COUNT = 4
UNCLEAR = "לא ברור"


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
OUTPUT_FIELDS = ["PatId", "treatment_type"] + [f"reason_{n}" for n in range(1, REASON_COUNT + 1)] + ["שגיאה"]
UNCATEGORIZED_CSV_PATH = "./uncategorized.csv"
OUTPUT_CSV_PATH = "./analysis_v1_results.csv"
NOT_IN_TOP_20_CSV_PATH = "./not_in_top_20.csv"
ANONYMOUS_CSV_PATH = "./analysis_v1_results_anonymous.csv"

# Hebrew system prompt for patient classification
SYSTEM_PROMPT = (
    "אתה רופא אונקולוג מומחה. עליך לבחון תיק מטופל ולקבוע את סוג הטיפול שהמטופל קיבל "
    "על סמך NCCN וESMO גידלינים. "
    "התבסס אך ורק על הטקסט של המטופל הזה. "
    "עליך להבחין בין שלושה סוגי טיפול בלבד: "
    "1. כימותרפיה ואימונותרפיה "
    "2. אימונותרפיה בלבד "
    "3. אימונותרפיה וכימותרפיה במינון מופחת "
    "אם הכימותרפיה ניתנה או הומלצה במינון מופחת, בחר בסוג 3. "
    "בנוסף, ציין 4 סיבות שבגללן נבחר סוג הטיפול, מהמשמעותית ביותר ועד הפחות משמעותית. "
    "כל סיבה בעד 3 מילים. "
    "אם אינך יכול לקבוע את סוג הטיפול או סיבה, כתוב 'לא ברור' במקומם, "
    "והסבר בשורת 'שגיאה' מה לא היה ברור. "
    "ענה בעברית בלבד.\n"
    "פורמט התשובה:\n"
    "סוג טיפול: [אחד משלושת הסוגים, או לא ברור]\n"
    "סיבה 1: [הסיבה המשמעותית ביותר]\n"
    "סיבה 2: [סיבה]\n"
    "סיבה 3: [סיבה]\n"
    "סיבה 4: [הסיבה הפחות משמעותית]\n"
    "שגיאה: [רק אם משהו לא ברור, אחרת השאר ריק]"
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
    """Move the last run's output files aside, then create a fresh output CSV with headers."""
    try:
        # Keep old results under a timestamped name so a new run never mixes with them
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for path in map(Path, (OUTPUT_CSV_PATH, UNCATEGORIZED_CSV_PATH, NOT_IN_TOP_20_CSV_PATH, ANONYMOUS_CSV_PATH)):
            if path.exists():
                old = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
                path.rename(old)
                print(f"Moved previous {path.name} to {old.name}")

        # Create file with headers
        with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
            csv.writer(csvfile).writerow(OUTPUT_FIELDS)
        print(f"Created output file: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to initialize output CSV: {e}")
        sys.exit(1)


def save_patient_result(pat_id: str, treatment_type: str, reasons: list[str], error: str) -> None:
    """Save patient result to output CSV file incrementally."""
    try:
        with open(OUTPUT_CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
            csv.writer(csvfile).writerow([pat_id, treatment_type] + reasons + [error])
    except Exception as e:
        print(f"  ERROR: Failed to write patient result: {e}")


def normalize_treatment_type(raw_treatment: str) -> str:
    """
    Normalize treatment type to standard English strings.

    Checks reduced dose first, then immuno only, then chemo + immuno, so a reduced
    dose answer is never read as plain chemo + immuno. A bare number 1-3 maps to
    the prompt's numbered list. Returns the raw value if nothing matches.
    """
    raw = raw_treatment.lower().strip()
    if not raw or UNCLEAR in raw:
        return UNCLEAR
    if "מופחת" in raw or "reduce" in raw:
        return TREATMENT_IMMUNO_CHEMO_REDUCED
    if "בלבד" in raw or "only" in raw or raw.startswith("רק "):
        return TREATMENT_IMMUNO_ONLY
    if ("כימו" in raw or "chemo" in raw) and ("אימונו" in raw or "immuno" in raw):
        return TREATMENT_CHEMO_IMMUNO
    by_number = {"1": TREATMENT_CHEMO_IMMUNO, "2": TREATMENT_IMMUNO_ONLY, "3": TREATMENT_IMMUNO_CHEMO_REDUCED}
    return by_number.get(raw[0], raw_treatment)


def parse_reasons(text: str) -> list[str]:
    """Read lines like 'סיבה 2: ...' into a list of REASON_COUNT reasons ('' where missing)."""
    reasons = [""] * REASON_COUNT
    for line in text.splitlines():
        match = re.match(r"^(?:סיבה|reason)\s*(\d)\s*[:：]\s*(.*)$", line.replace("*", "").strip(), re.IGNORECASE)
        if match and 1 <= int(match.group(1)) <= REASON_COUNT:
            reasons[int(match.group(1)) - 1] = match.group(2).strip()
    return reasons


def reasons_ok(reasons: list[str]) -> bool:
    """True when every reason is there and has at most MAX_REASON_WORDS words."""
    return all(r and len(r.split()) <= MAX_REASON_WORDS for r in reasons)


def parse_llm_response(pat_id: str, llm_response: str) -> tuple[str, list[str], str]:
    """
    Parse LLM response into (treatment_type, reasons, error).

    A treatment type that is not one of the three becomes 'לא ברור', with the
    model's answer in error and a row in uncategorized.csv.
    """
    raw_treatment_type = ""
    error = ""
    for line in llm_response.splitlines():
        line = line.replace("*", "").strip()
        label, _, value = line.partition(":")
        label = label.strip()
        if label in ("סוג טיפול", "טיפול") or label.lower() == "treatment type":
            raw_treatment_type = value.strip()
        elif label in ("שגיאה", "error"):
            error = value.strip()

    treatment_type = normalize_treatment_type(raw_treatment_type)
    if treatment_type not in (TREATMENT_CHEMO_IMMUNO, TREATMENT_IMMUNO_ONLY, TREATMENT_IMMUNO_CHEMO_REDUCED, UNCLEAR):
        print(f"  WARNING: Treatment type '{raw_treatment_type}' is not one of the three types")
        write_uncategorized_patient(pat_id, llm_response, UNCLEAR)
        error = error or f"סוג הטיפול שהמודל כתב אינו אחד משלושת הסוגים: {raw_treatment_type}"
        treatment_type = UNCLEAR
    reasons = parse_reasons(llm_response)
    if (treatment_type == UNCLEAR or UNCLEAR in reasons) and not error:
        error = "המודל כתב 'לא ברור' ולא הסביר למה"
    print(f"  Treatment type: {treatment_type}")

    return treatment_type, reasons, error


def classify_patient(
    pat_id: str,
    current_disease: str,
    summary_conclusions: str,
    recommendations: str,
    provider: str,
) -> tuple[bool, str, str, str]:
    """
    Classify a patient into treatment type using LLM.

    Returns tuple of (is_error, treatment_type, primary_reason, raw_response).
    If LLM returns error, returns True for is_error.
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

    # Check for LLM API errors
    if response.startswith("ERROR:"):
        print(f"  ERROR: LLM call failed: {response}")
        return True, "ERROR", response, response

    return False, "", "", response

def fix_reasons(llm_response: str, reasons: list[str], provider: str) -> tuple[list[str], str]:
    """
    Ask the model to sum up each reason in at most MAX_REASON_WORDS words, up to MAX_REASON_TRIES times.

    Returns (reasons, error). A reason still missing at the end becomes 'לא ברור', and error says why.
    """
    for _ in range(MAX_REASON_TRIES):
        if reasons_ok(reasons):
            return reasons, ""
        if provider != "anthropic":
            time.sleep(REQUEST_DELAY_SECONDS)
        response = invoke_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt_text=(
                f"תשובתך הקודמת:\n{llm_response}\n\n"
                f"סכם כל אחת מ-{REASON_COUNT} הסיבות לסיבה של עד {MAX_REASON_WORDS} מילים, "
                "באותו סדר. אם סיבה חסרה, כתוב 'לא ברור'. "
                "ענה רק בשורות 'סיבה 1:' עד 'סיבה 4:'."
            ),
            max_tokens=1000,
            temperature=0.0,
            provider_override=provider,
        )
        if response.startswith("ERROR:"):
            print(f"  ERROR: Retry for reasons failed: {response}")
            continue
        reasons = [new or old for new, old in zip(parse_reasons(response), reasons)]
        print(f"  Reasons after summing up: {reasons}")
    if reasons_ok(reasons):
        return reasons, ""
    print(f"  WARNING: Reasons still missing or over {MAX_REASON_WORDS} words after {MAX_REASON_TRIES} tries")
    missing = [n for n, r in enumerate(reasons, 1) if not r]
    error = f"המודל לא נתן את סיבות {missing}" if missing else ""
    return [r or UNCLEAR for r in reasons], error


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


def merge_similar_reasons(reasons: list[str], provider: str) -> dict[str, str]:
    """Ask the model to map each distinct reason to a group name. Returns {reason: group}."""
    if provider != "anthropic":
        time.sleep(REQUEST_DELAY_SECONDS)
    response = invoke_llm(
        system_prompt=(
            "אתה רופא אונקולוג מומחה. להלן רשימת סיבות להחלטות טיפוליות. "
            f"מזג סיבות בעלות אותה משמעות תחת שם קבוצה אחד של עד {MAX_REASON_WORDS} מילים. "
            "לכל סיבה ברשימה כתוב שורה אחת, ללא מספור, בפורמט: סיבה => שם קבוצה. "
            "ענה בעברית בלבד."
        ),
        user_prompt_text="\n".join(sorted(set(reasons))),
        max_tokens=8000,
        temperature=0.0,
        provider_override=provider,
    )
    if response.startswith("ERROR:"):
        print(f"  ERROR: Merging reasons failed, counting exact wording: {response}")
        return {}
    groups = {}
    for line in response.splitlines():
        if "=>" in line:
            reason, group = line.split("=>", 1)
            groups[reason.strip(" -*•")] = group.strip()
    return groups


def append_top_20_by_treatment(
    patient_results: list[tuple[str, str]], main_results: list[tuple[str, str]], provider: str
) -> None:
    """
    Append two sets of rows to the output CSV, one row per treatment type in each:
    the 20 most common reasons over all four reasons, then over the most important reason only.
    Both sets share one grouping of similar reasons per treatment type.
    """
    all_rows, main_rows = [], []
    for treatment in (TREATMENT_CHEMO_IMMUNO, TREATMENT_IMMUNO_ONLY, TREATMENT_IMMUNO_CHEMO_REDUCED):
        reasons = [r for t, r in patient_results if t == treatment and r]
        if not reasons:
            continue
        main_reasons = [r for t, r in main_results if t == treatment and r]
        print(f"\n--- Top 20 reasons for {treatment} ({len(reasons)} reasons, {len(main_reasons)} main) ---")
        groups = merge_similar_reasons(reasons, provider)
        for rows, label, items in ((all_rows, "all reasons", reasons), (main_rows, "main reason", main_reasons)):
            # ponytail: a reason the model left out or reworded keeps its own name
            top = Counter(groups.get(r, r) for r in items).most_common(20)
            print(f"  {label}: " + ", ".join(f"{name} ({count})" for name, count in top))
            rows.append([f"TOP_20 {label}: {treatment}"] + [f"{name} ({count})" for name, count in top])
    try:
        with open(OUTPUT_CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
            csv.writer(csvfile).writerows(all_rows + main_rows)
        print(f"Appended top 20 reasons per treatment type to {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to append top 20 reasons per treatment type: {e}")


def write_anonymous_csv() -> None:
    """
    Copy the output CSV without the PatId column. Patient rows go in random order,
    so nobody can match them back to cases.csv by position. The overall top 20
    explanations may name patients, so the copy leaves them out.
    """
    try:
        with open(OUTPUT_CSV_PATH, "r", newline="", encoding="utf-8") as csvfile:
            header, *rows = list(csv.reader(csvfile))
        patients = [row[1:] for row in rows if not row[0].startswith("TOP_")]
        random.shuffle(patients)
        tops = [row[:3] if row[0].startswith("TOP_REASON_") else row for row in rows if row[0].startswith("TOP_")]
        with open(ANONYMOUS_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header[1:])
            writer.writerows(patients + tops)
        print(f"Created {ANONYMOUS_CSV_PATH} without patient IDs")
    except Exception as e:
        print(f"ERROR: Failed to create anonymous CSV: {e}")


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
    # (treatment type, reason) for each of each patient's reasons, for the top 20 per treatment type
    patient_results: list[tuple[str, str]] = []
    # (treatment type, most important reason) per patient, for the second top 20 set
    main_results: list[tuple[str, str]] = []

    # Track patients by treatment type for validation
    treatment_counts: dict[str, int] = {
        TREATMENT_CHEMO_IMMUNO: 0,
        TREATMENT_IMMUNO_ONLY: 0,
        TREATMENT_IMMUNO_CHEMO_REDUCED: 0,
        UNCLEAR: 0,
    }

    # Track LLM API errors
    error_count = 0

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
        is_error, _, _, llm_response = classify_patient(
            pat_id=pat_id,
            current_disease=current_disease,
            summary_conclusions=summary_conclusions,
            recommendations=recommendations,
            provider=args.provider,
        )

        # Handle LLM API errors
        if is_error:
            error_count += 1
            treatment_type, reasons, error = UNCLEAR, [UNCLEAR] * REASON_COUNT, llm_response
        else:
            treatment_type, reasons, error = parse_llm_response(pat_id, llm_response)
            if not reasons_ok(reasons):
                print(f"  Reasons missing or over {MAX_REASON_WORDS} words, asking to sum them up...")
                reasons, reasons_error = fix_reasons(llm_response, reasons, args.provider)
                error = "; ".join(e for e in (error, reasons_error) if e)
            print(f"  Reasons: {reasons}")

        save_patient_result(pat_id, treatment_type, reasons, error)
        treatment_counts[treatment_type] += 1
        # The overall top 20 step works on each patient's most important reason
        all_patient_reasons.append(PatientReason(pat_id=pat_id, reason_text=reasons[0]))
        patient_results += [(treatment_type, r) for r in reasons if r != UNCLEAR]
        if reasons[0] != UNCLEAR:
            main_results.append((treatment_type, reasons[0]))

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

    # Print error summary
    print(f"\n--- Error Summary ---")
    print(f"Total LLM API errors: {error_count}")
    if error_count > 0:
        print(f"Patients with errors: {error_count}")
    else:
        print("No LLM API errors encountered")

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

        append_top_20_by_treatment(patient_results, main_results, args.provider)
    else:
        print("\nWARNING: No patient reasons collected, skipping top-20 analysis")

    write_anonymous_csv()


if __name__ == "__main__":
    main()
