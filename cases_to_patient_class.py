import json
import re
import sys
import csv
import time
import os
from llm_client import invoke_llm  # Import the new common function

# Configs
REQUEST_DELAY_SECONDS = 31  # Delay in seconds between requests (current rate limit is 2 req/sec)

# Model ID for Claude 3.5 Sonnet on Bedrock (as used in this script)
# This specific ID will be passed to the invoke_llm function.
BEDROCK_CLAUDE_MODEL_ID = "eu.anthropic.claude-3-5-sonnet-20240620-v1:0"


class Patient:
    def __init__(
        self,
        cancer_type: str,
        metastasized: bool | None,
        age: int | None,
        background_illnesses: list[str],
        treatment_type: str,
        reason_for_treatment: str,
        pdl1_score: float | None,
        dosage_change: float | None,
        chemotherapy_medication_type: str | None,
    ):
        self.cancer_type = cancer_type
        self.metastasized = metastasized
        self.age = age
        self.background_illnesses = background_illnesses
        self.treatment_type = treatment_type
        self.reason_for_treatment = reason_for_treatment
        self.pdl1_score = pdl1_score
        self.dosage_change = dosage_change
        self.chemotherapy_medication_type = chemotherapy_medication_type

    def __repr__(self):
        return (
            f"Patient(cancer_type='{self.cancer_type}', "
            f"metastasized={self.metastasized}, age={self.age}, "
            f"background_illnesses={self.background_illnesses}, "
            f"treatment_type='{self.treatment_type}', "
            f"reason_for_treatment='{self.reason_for_treatment}', "
            f"pdl1_score={self.pdl1_score}, "
            f"dosage_change={self.dosage_change}, "
            f"chemotherapy_medication_type='{self.chemotherapy_medication_type}')"
        )


# Define the constant system prompt for patient data extraction
SYSTEM_PROMPT_PATIENT_EXTRACTION_EN = (
    "You are an AI assistant specialized in extracting structured information from oncological medical case notes.\n"
    "Given a patient's case information (Current Disease, Summary & Conclusions, Recommendations), extract the following details:\n"
    "1.  **Cancer Type**: (String, e.g., \"Lung Cancer\", \"Breast Cancer\", \"Prostate Cancer\". If not determinable, use \"Unknown\")\n"
    "2.  **Metastasized**: (Boolean: True if cancer has metastasized or is described as advanced/spread, False otherwise or if localized. If not determinable, use null.)\n"
    "3.  **Age**: (Integer: Patient's age in years. If not determinable, use null.)\n"
    "4.  **Background Illnesses**: (List of strings: e.g., [\"Diabetes Type 2\", \"Hypertension\"]. If none mentioned or not determinable, use an empty list.)\n"
    "5.  **Treatment Type**: (String: Must be one of \"Immunotherapy and Chemotherapy\" or \"Immunotherapy Only\". If the text suggests a different primary oncological treatment (e.g. chemo only, surgery only, radiation only) or if it's unclear, state \"Other/Unclear\".)\n"
    "6.  **Reason for Treatment Choice**: (String: Briefly explain the rationale for the chosen treatment type based on the provided text. e.g., \"High PD-L1 expression led to immunotherapy alone\", \"Advanced stage with nodal involvement prompted combination therapy\". If not determinable, use \"Not Specified\".)\n"
    "7.  **PDL1 Score**: (Float between 0.0 and 1.0. If a percentage is mentioned (e.g., \"PD-L1 50%\", \"PDL1 >50%\", \"PDL1 <1%\"), convert it to a decimal (e.g., 0.5 for 50%, 0.5 for >50% if no more specific value, 0.01 for <1% if no more specific value). If a general term like \"high\" or \"low\" is used without a number, or if testing is recommended but no result given, use null. If not mentioned at all, use null.)\n"
    "8.  **Dosage Change**: (Float: Applicable only if 'Treatment Type' is \"Immunotherapy and Chemotherapy\". If a dosage change (e.g., reduction by 20%, increase by 10%) for chemotherapy is mentioned, provide the percentage of change as a float (e.g., -20.0 for a 20% reduction, 10.0 for a 10% increase). If no change is mentioned or it's explicitly stated as no change, use 0.0. If a change is mentioned but not quantifiable (e.g., \"dose adjustment\"), use -1.0 (representing 'unquantifiable change'). If not applicable (e.g. treatment is 'Immunotherapy Only' or 'Other/Unclear'), use null.)\n"
    "9.  **Chemotherapy Medication Type**: (String: Applicable only if 'Treatment Type' is \"Immunotherapy and Chemotherapy\". List the type(s) of chemotherapy medication mentioned, e.g., \"R-CHOP\", \"Paclitaxel\". If multiple, join with a comma. If not applicable or not mentioned, use \"N/A\".)\n\n"
    "Provide the output as a JSON object with keys: \"cancer_type\", \"metastasized\", \"age\", \"background_illnesses\", \"treatment_type\", \"reason_for_treatment_choice\", \"pdl1_score\", \"dosage_change\", \"chemotherapy_medication_type\".\n"
    "Ensure all string fields are populated, using \"Not Specified\", \"N/A\", or \"Unknown\" where appropriate if information cannot be extracted. For lists, use an empty list if no information. For numbers/booleans, use null if not determinable.\n"
)


input_csv_path = './cases.csv'
ORIGINAL_FIELDNAMES = ['Current_Disease', 'Summary_Conclusions', 'Recommendations']

try:
    with open(input_csv_path, 'r', encoding='utf-8') as infile:
        patients_list = []
        reader = csv.DictReader(infile)
        # Ensure the reader uses the correct fieldnames if they are not exactly as expected
        if reader.fieldnames != ORIGINAL_FIELDNAMES:
            print(
                f"WARNING: CSV headers in '{input_csv_path}' are {reader.fieldnames}, expected {ORIGINAL_FIELDNAMES}."
            )
            sys.exit(1)

        for i, row in enumerate(reader):
            if i > 0:  # If it's not the first record, wait before processing this new record
                print(f"\n--- Waiting {REQUEST_DELAY_SECONDS} seconds before processing record {i+1}... ---")
                time.sleep(REQUEST_DELAY_SECONDS)  # Respect rate limits

            print(f"\n\n--- Processing record {i+1} from CSV ---")

            current_disease_text = row.get(ORIGINAL_FIELDNAMES[0], "")
            doctor_summary_text = row.get(ORIGINAL_FIELDNAMES[1], "")
            doctor_recommendations_text = row.get(ORIGINAL_FIELDNAMES[2], "")

            # Prepare user prompt for LLM
            user_prompt_for_extraction = f"""
Case Information:

Current Disease:
{current_disease_text}

Summary & Conclusions (Doctor):
{doctor_summary_text}

Recommendations (Doctor):
{doctor_recommendations_text}

---
Please extract the patient details based on the above information and provide a JSON output.
"""

            print(f"--- Invoking LLM for patient data extraction (record {i+1}) ---")

            llm_response_text = invoke_llm(
                system_prompt=SYSTEM_PROMPT_PATIENT_EXTRACTION_EN,
                user_prompt_text=user_prompt_for_extraction,
                max_tokens=2000,  # Increased max_tokens for potentially complex extractions
                temperature=0.0,
            )

            if llm_response_text.startswith("ERROR:"):
                print(f"ERROR during LLM call for patient data extraction (record {i+1}): {llm_response_text}")
                # Create a patient object with error indicators or skip
                patient = Patient(
                    cancer_type="Error: LLM Failed",
                    metastasized=None,
                    age=None,
                    background_illnesses=[],
                    treatment_type="Error: LLM Failed",
                    reason_for_treatment="Error: LLM Failed",
                    pdl1_score=None,
                    dosage_change=None,
                    chemotherapy_medication_type="Error: LLM Failed",
                )
            else:
                try:
                    extracted_data = json.loads(llm_response_text)

                    treatment_type = extracted_data.get("treatment_type", "Other/Unclear")
                    if treatment_type not in ["Immunotherapy and Chemotherapy", "Immunotherapy Only"]:
                        print(
                            f"WARNING (Record {i+1}): LLM returned treatment_type '{treatment_type}'. Expected 'Immunotherapy and Chemotherapy' or 'Immunotherapy Only'. Storing as is.",
                            file=sys.stderr,
                        )
                        # No need to change it if LLM was instructed to use "Other/Unclear"

                    patient = Patient(
                        cancer_type=extracted_data.get("cancer_type", "Unknown"),
                        metastasized=extracted_data.get("metastasized"),  # None if missing
                        age=extracted_data.get("age"),  # None if missing
                        background_illnesses=extracted_data.get("background_illnesses", []),
                        treatment_type=treatment_type,
                        reason_for_treatment=extracted_data.get("reason_for_treatment_choice", "Not Specified"),
                        pdl1_score=extracted_data.get("pdl1_score"),  # None if missing
                        dosage_change=extracted_data.get("dosage_change"),  # None if missing or not applicable
                        chemotherapy_medication_type=extracted_data.get("chemotherapy_medication_type", "N/A"),
                    )
                except json.JSONDecodeError:
                    print(
                        f"ERROR: Could not parse JSON response from LLM for record {i+1}. Response: {llm_response_text}"
                    )
                    patient = Patient(
                        cancer_type="Error: JSON Parse Failed",
                        metastasized=None,
                        age=None,
                        background_illnesses=[],
                        treatment_type="Error: JSON Parse Failed",
                        reason_for_treatment="Error: JSON Parse Failed",
                        pdl1_score=None,
                        dosage_change=None,
                        chemotherapy_medication_type="Error: JSON Parse Failed",
                    )
            patients_list.append(patient)
            print(f"--- Finished processing record {i+1}. Patient object created. ---")

        print("\n\n--- List of Extracted Patients ---")
        for patient_obj in patients_list:
            print(patient_obj)

except FileNotFoundError:
    print(f"ERROR: Input CSV file '{input_csv_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

print("\n\n--- Script Finished ---")
