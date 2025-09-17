import json
import re
import sys
import csv
import time
import os
from llm_client import invoke_llm  # Import the new common function

def extract_4_reasons(llm_response_text):
    """Extract 4 numbered reasons from LLM response text"""
    reasons = ["", "", "", ""]

    # Split by lines and look for numbered items
    lines = llm_response_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        # Look for patterns like "1. reason", "1) reason", etc.
        for i in range(1, 6):
            patterns = [f"{i}. ", f"{i}) ", f"{i}- "]
            for pattern in patterns:
                if line.startswith(pattern):
                    reason_text = line[len(pattern):].strip()
                    if reason_text:  # Only update if we found actual content
                        reasons[i-1] = reason_text
                    break

    return reasons

# Configs
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure_openai").lower()
REQUEST_DELAY_SECONDS = 31  # Delay in seconds between requests (current rate limit is 2 req/sec)
ANTHROPIC_NO_RATE_LIMIT = LLM_PROVIDER == "anthropic"

# Define the constant system prompt for treatment plan
SYSTEM_PROMPT_BASE_HE = (
    "אתה רופא אונקולוג עלייך לבסס את התשובות שלך על בסיס NCCN וESNO , "
    "אתה צריך לציין את הסיבות וההגיון שהובילו את הרופא להחלטה על טיפול "
    "כל מטופל מקבל טיפול אחד משני סוגים או רק אימונו או אימונו וכימו, עלייך להבין איזה סוג טיפול המטופל קיבל"
    " ולדרג את השיקולים שלו לפי הסדר , לסדר את זה בצורה מדורגת לפי עוצמה שהשפיעה על החלטת הטיפול בין " \
    "אם זה אימונולוגי לבד או אימונולוגי וכימו. "
    "ענה בעברית בלבד ורשום בדיוק 4 סיבות "
    "הסיבה צריכה להיבחר מהרשימה הבאה:"
    "PS Good 0-1,"
    "PS Intermediate 2,"
    "Age young ,"
    "Age old ,"
    "PDL-1 high,"
    "PDL-1 low,"
    "PDL-1 unknown,"
    "High disease burden,"
    "low disease burden,"
    "Comorbidities renal,"
    "Comorbidities cardiac,"
    "Comorbidities hepatic,"
    "Comorbidities pulmonary\copd,"
    "Comorbidities autoimmune,"
    "Comorbidities viral(HBV\HIV),"
    "Comorbidities other,"
    "PDL-1 low,"
    "PDL-1 low,"
    "Curative,"
    "Palliative,"
    "QoL priority,"
    "Refusal of chemo,"
    "Awaiting NGS,"
    "Dx not final,"
    "Material insufficient,"
    " הכי חשובות בפורמט הבא: "
    "1. [סוג טיפול]"
    "2. [סיבה ראשונה] "
    "3. [סיבה שנייה] "
    "4. [סיבה שלישית] "
    "5. [סיבה רביעית]"
)



input_csv_path = './cases.csv'
output_csv_path = 'analysis_per_case.csv'
DEFAULT_SCORE_ON_ERROR = 0.0
CSV_FIELD_DISEASE = 'Current_Disease'
CSV_FIELD_SUMMARY_CONCLUSION = 'Summary_Conclusions'
CSV_FIELD_RECOMMENDATIONS = 'Recommendations'
ORIGINAL_FIELDNAMES = ['PatId', 'Current_Disease', 'Summary_Conclusions', 'Recommendations']
NEW_FIELDNAMES = ['treatment_type','reason_1', 'reason_2', 'reason_3', 'reason_4']
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

        reasons_dic = []

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
            user_prompt+= "זה מה שהחליט הרופא:\n" + doctor_recommendations_text + " " + doctor_summary_text

            reasons = ["Error: LLM call failed or no content.", "", "", ""]
           
            # 1. First LLM Call: Get summary/conclusion resoning
            print("--- Invoking LLM for treatment reasoning ---")

            llm_response_text = invoke_llm(
                system_prompt=SYSTEM_PROMPT_BASE_HE,
                user_prompt_text=user_prompt,
                max_tokens=1000,
                temperature=0.0,
                # provider_override can be used here if needed, e.g., os.getenv("LLM_PROVIDER_CASES", "bedrock")
            )

            if llm_response_text.startswith("ERROR:"):
                print(f"ERROR during LLM call for treatment plan (record {i+1}): {llm_response_text}")
                # reasons remains with error message as default
            else:
                reasons = extract_4_reasons(llm_response_text)

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
                'treatment_type': reasons[0],
                'reason_1': reasons[1],
                'reason_2': reasons[2],
                'reason_3': reasons[3],
                'reason_4': reasons[4],
            }
            writer.writerow(output_row)
            print(f"--- Finished processing and wrote record {i+1} to '{output_csv_path}' ---")
            reasons_dic += reasons[1:5]

except FileNotFoundError:
    print(f"ERROR: Input CSV file '{input_csv_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

print ("\n\n reasons list: \n\n "+"\n".join(reasons_dic))
print("\n\n--- Script Finished ---")
