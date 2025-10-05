import json
import re
import sys
import csv
import time
import os
from llm_client import invoke_llm  # Import the new common function


def extract_reason_percentages(llm_response_text):
    """Extract treatment type and percentage values for all reasons from LLM response text"""

    # Define all possible reasons
    all_reasons = [
        "PS Good 0-1",
        "PS Intermediate 2",
        "PS Bad 3-4",
        "Age young",
        "Age old",
        "PDL-1 high",
        "PDL-1 low",
        "PDL-1 unknown",
        "High disease burden",
        "low disease burden",
        "Comorbidities renal",
        "Comorbidities cardiac",
        "Comorbidities hepatic",
        "Comorbidities pulmonary/copd",
        "Comorbidities autoimmune",
        "Comorbidities viral(HBV/HIV)",
        "Comorbidities other",
        "Curative",
        "Palliative",
        "QoL priority",
        "Refusal of chemo",
        "Awaiting NGS",
        "Dx not final",
        "Material insufficient",
    ]

    # Initialize result dictionary
    result = {'treatment_type': '', 'reasons': {reason: 0.0 for reason in all_reasons}}

    lines = llm_response_text.strip().split('\n')

    for line in lines:
        line = line.strip()

        # Extract treatment type
        if line.startswith("סוג טיפול:"):
            result['treatment_type'] = line.replace("סוג טיפול:", "").strip()
            continue

        # Extract percentage for each reason
        for reason in all_reasons:
            pattern = f"{reason}:"
            if line.startswith(pattern):
                # Extract percentage value
                percentage_text = line.replace(pattern, "").strip()
                # Remove % sign if present and convert to float
                try:
                    percentage_value = float(percentage_text.replace('%', '').strip())
                    result['reasons'][reason] = percentage_value
                except ValueError:
                    # If conversion fails, set to 0
                    result['reasons'][reason] = 0.0
                break

    return result


# Configs
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure_openai").lower()
REQUEST_DELAY_SECONDS = 31  # Delay in seconds between requests (current rate limit is 2 req/sec)
ANTHROPIC_NO_RATE_LIMIT = LLM_PROVIDER == "anthropic"

# Define the constant system prompt for treatment plan
SYSTEM_PROMPT_BASE_HE = (
    "אתה רופא אונקולוג עלייך לבסס את התשובות שלך על בסיס NCCN וESMO , "
    "אתה צריך לציין את הסיבות וההגיון שהובילו את הרופא להחלטה על טיפול "
    "כל מטופל מקבל טיפול אחד משני סוגים או רק אימונו או אימונו וכימו, עלייך להבין איזה סוג טיפול המטופל קיבל"
    " ולדרג את השיקולים שלו לפי הסדר , לסדר את זה בצורה מדורגת לפי עוצמה שהשפיעה על החלטת הטיפול בין "
    "  אם זה 'אימונותרפיה בלבד' או 'אימונותרפיה וכימותרפיה' או מינון מופחת של כימותרפיה. "
    "ענה מהרשימה בלבד ורשום את אחוז ההשפעה של כל סיבה, כאשר סכום כל האחוזים צריך להיות 100"
    "הסיבות צריכות להיבחר מהרשימה הבאה:"
    "PS Good 0-1"
    "PS Intermediate 2"
    "PS Bad 3-4"
    "Age young"
    "Age old"
    "PDL-1 high"
    "PDL-1 low"
    "PDL-1 unknown"
    "High disease burden"
    "low disease burden"
    "Comorbidities renal"
    "Comorbidities cardiac"
    "Comorbidities hepatic"
    "Comorbidities pulmonary/copd"
    "Comorbidities autoimmune"
    "Comorbidities viral(HBV/HIV)"
    "Comorbidities other"
    "Curative"
    "Palliative"
    "QoL priority"
    "Refusal of chemo"
    "Awaiting NGS"
    "Dx not final"
    "Material insufficient"
    ""
    "ענה בפורמט הבא בלבד:"
    "סוג טיפול: [סוג טיפול]"
    "ואז עבור כל סיבה מהרשימה, ציין את אחוז ההשפעה שלה על ההחלטה, כאשר סכום כל האחוזים הוא 100:"
    "PS Good 0-1: [אחוז]"
    "PS Intermediate 2: [אחוז]"
    "PS Bad 3-4: [אחוז]"
    "Age young: [אחוז]"
    "Age old: [אחוז]"
    "PDL-1 high: [אחוז]"
    "PDL-1 low: [אחוז]"
    "PDL-1 unknown: [אחוז]"
    "High disease burden: [אחוז]"
    "low disease burden: [אחוז]"
    "Comorbidities renal: [אחוז]"
    "Comorbidities cardiac: [אחוז]"
    "Comorbidities hepatic: [אחוז]"
    "Comorbidities pulmonary/copd: [אחוז]"
    "Comorbidities autoimmune: [אחוז]"
    "Comorbidities viral(HBV/HIV): [אחוז]"
    "Comorbidities other: [אחוז]"
    "Curative: [אחוז]"
    "Palliative: [אחוז]"
    "QoL priority: [אחוז]"
    "Refusal of chemo: [אחוז]"
    "Awaiting NGS: [אחוז]"
    "Dx not final: [אחוז]"
    "Material insufficient: [אחוז]"
)


input_csv_path = './cases.csv'
output_csv_path = 'analysis_per_case.csv'
DEFAULT_SCORE_ON_ERROR = 0.0
CSV_FIELD_DISEASE = 'Current_Disease'
CSV_FIELD_SUMMARY_CONCLUSION = 'Summary_Conclusions'
CSV_FIELD_RECOMMENDATIONS = 'Recommendations'
ORIGINAL_FIELDNAMES = ['PatId', 'Current_Disease', 'Summary_Conclusions', 'Recommendations']

# Define all reason columns
ALL_REASONS = [
    "PS Good 0-1",
    "PS Intermediate 2",
    "PS Bad 3-4",
    "Age young",
    "Age old",
    "PDL-1 high",
    "PDL-1 low",
    "PDL-1 unknown",
    "High disease burden",
    "low disease burden",
    "Comorbidities renal",
    "Comorbidities cardiac",
    "Comorbidities hepatic",
    "Comorbidities pulmonary/copd",
    "Comorbidities autoimmune",
    "Comorbidities viral(HBV/HIV)",
    "Comorbidities other",
    "Curative",
    "Palliative",
    "QoL priority",
    "Refusal of chemo",
    "Awaiting NGS",
    "Dx not final",
    "Material insufficient",
]

# Create column names by replacing spaces with underscores
REASON_COLUMNS = [
    reason.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') for reason in ALL_REASONS
]
NEW_FIELDNAMES = ['treatment_type'] + REASON_COLUMNS
output_fieldnames = ORIGINAL_FIELDNAMES + NEW_FIELDNAMES

try:
    with open(input_csv_path, 'r', encoding='utf-8') as infile, open(
        output_csv_path, 'w', newline='', encoding='utf-8'
    ) as outfile:

        reader = csv.DictReader(infile)
        # Ensure the reader uses the correct fieldnames if they are not exactly as expected
        if reader.fieldnames != ORIGINAL_FIELDNAMES:
            print(
                f"WARNING: CSV headers in '{input_csv_path}' are {reader.fieldnames}, expected {ORIGINAL_FIELDNAMES}."
            )
        # Check if essential columns are present
        if not reader.fieldnames or not (
            CSV_FIELD_DISEASE in reader.fieldnames
            and CSV_FIELD_SUMMARY_CONCLUSION in reader.fieldnames
            and CSV_FIELD_RECOMMENDATIONS in reader.fieldnames
        ):
            print(
                f"ERROR: Essential columns ('{CSV_FIELD_DISEASE}', '{CSV_FIELD_SUMMARY_CONCLUSION}', '{CSV_FIELD_RECOMMENDATIONS}') not found in CSV headers. Exiting."
            )
            sys.exit(1)

        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            if (
                i > 0 and not ANTHROPIC_NO_RATE_LIMIT
            ):  # If it's not the first record, wait before processing this new record
                print(f"\n--- Waiting {REQUEST_DELAY_SECONDS} seconds before processing record {i+1}... ---")
                time.sleep(REQUEST_DELAY_SECONDS)  # Respect rate limits

            print(f"\n\n--- Processing record {i+1} from CSV ---")

            pat_id = row.get('PatId', "")
            current_disease_text = row.get('Current_Disease', "")
            doctor_summary_text = row.get('Summary_Conclusions', "")
            doctor_recommendations_text = row.get('Recommendations', "")

            user_prompt = "כך סיכם הרופא את המקרה:\n" + current_disease_text + "\n\n"
            user_prompt += "זה מה שהחליט הרופא:\n" + doctor_recommendations_text + " " + doctor_summary_text

            # Initialize result with default values
            analysis_result = {
                'treatment_type': 'Error: LLM call failed or no content.',
                'reasons': {reason: 0.0 for reason in ALL_REASONS},
            }

            # 1. First LLM Call: Get summary/conclusion reasoning
            print("--- Invoking LLM for treatment reasoning ---")

            llm_response_text = invoke_llm(
                system_prompt=SYSTEM_PROMPT_BASE_HE,
                user_prompt_text=user_prompt,
                max_tokens=1500,
                temperature=0.0,
                # provider_override can be used here if needed, e.g., os.getenv("LLM_PROVIDER_CASES", "bedrock")
            )

            if llm_response_text.startswith("ERROR:"):
                print(f"ERROR during LLM call for treatment plan (record {i+1}): {llm_response_text}")
                # analysis_result remains with error message as default
            else:
                analysis_result = extract_reason_percentages(llm_response_text)

            if not ANTHROPIC_NO_RATE_LIMIT:
                # Wait before the second LLM request for the current record
                print(
                    f"\n--- Waiting {REQUEST_DELAY_SECONDS} seconds before AI vs Doctor comparison request for record {i+1}... ---"
                )
                time.sleep(REQUEST_DELAY_SECONDS)

            # Write data to output CSV
            output_row = {
                'PatId': pat_id,
                'Current_Disease': current_disease_text,
                'Summary_Conclusions': doctor_summary_text,
                'Recommendations': doctor_recommendations_text,
                'treatment_type': analysis_result['treatment_type'],
            }

            # Add all reason percentages to output row
            for i, reason in enumerate(ALL_REASONS):
                column_name = REASON_COLUMNS[i]
                output_row[column_name] = analysis_result['reasons'][reason]

            writer.writerow(output_row)
            print(f"--- Finished processing and wrote record {i+1} to '{output_csv_path}' ---")

except FileNotFoundError:
    print(f"ERROR: Input CSV file '{input_csv_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

print("\n\n--- Script Finished ---")
