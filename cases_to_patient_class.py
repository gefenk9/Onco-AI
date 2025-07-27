import json
import re
import sys
import csv
import time
import os
from collections import Counter  # Added for analysis
from dotenv import load_dotenv  # To read .env file for LLM_PROVIDER
from llm_client import invoke_llm  # Import the new common function

# Load environment variables from .env file if it exists, to check LLM_PROVIDER
load_dotenv()

# Configs
LLM_PROVIDER_CONFIG = os.getenv("LLM_PROVIDER", "bedrock").lower()
REQUEST_DELAY_SECONDS = 0 if LLM_PROVIDER_CONFIG == "anthropic" else 31

print(f"--- LLM Provider: {LLM_PROVIDER_CONFIG}, Request Delay: {REQUEST_DELAY_SECONDS}s ---")

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
    "Your entire response MUST be a single, valid JSON object. Do not include any explanatory text, markdown, or any characters outside of this JSON object.\n"
    "The JSON object must have the following keys: \"cancer_type\", \"metastasized\", \"age\", \"background_illnesses\", \"treatment_type\", \"reason_for_treatment_choice\", \"pdl1_score\", \"dosage_change\", \"chemotherapy_medication_type\".\n"
    "Ensure all string fields are populated, using \"Not Specified\", \"N/A\", or \"Unknown\" where appropriate if information cannot be extracted. For lists, use an empty list if no information. For numbers/booleans, use null if not determinable."
)


def perform_analysis_and_print_results(patients: list[Patient]):
    print("\n\n--- Performing Analyses ---")

    if not patients:
        print("No patients to analyze.")
        return

    # Filter patients
    immuno_only_patients = [p for p in patients if p.treatment_type == "Immunotherapy Only"]
    combo_patients = [p for p in patients if p.treatment_type == "Immunotherapy and Chemotherapy"]

    # 1. Average age of patients that only got immunotherapy
    # 5. Avg age of patients that got only immunotherapy (Points 1 and 5 are the same)
    print("\n--- Analysis 1 & 5: Average age of patients who received Immunotherapy Only ---")
    ages_immuno_only = [p.age for p in immuno_only_patients if p.age is not None]
    if ages_immuno_only:
        avg_age_immuno_only = sum(ages_immuno_only) / len(ages_immuno_only)
        print(f"Average age: {avg_age_immuno_only:.2f} years (based on {len(ages_immuno_only)} patients)")
    else:
        print("No patients with 'Immunotherapy Only' treatment and known age found.")

    # 2. Reasons for getting immunotherapy, sorted by most common first
    print("\n--- Analysis 2: Reasons for getting Immunotherapy Only (Most common first) ---")
    if immuno_only_patients:
        reasons_immuno_only = [p.reason_for_treatment for p in immuno_only_patients if p.reason_for_treatment]
        if reasons_immuno_only:
            reason_counts = Counter(reasons_immuno_only)
            print("Reasons:")
            for reason, count in reason_counts.most_common():
                print(f"- \"{reason}\": {count} occurrences")
        else:
            print("No reasons specified for 'Immunotherapy Only' patients.")
    else:
        print("No patients with 'Immunotherapy Only' treatment found.")

    # 3. Percentage of patients that got only immunotherapy and their PDL1 is lower than 0.5
    print("\n--- Analysis 3: Percentage of 'Immunotherapy Only' patients with PDL1 < 0.5 ---")
    if immuno_only_patients:
        # Ensure there are patients to avoid division by zero if the list is empty after filtering
        if len(immuno_only_patients) > 0:
            low_pdl1_immuno_only = [p for p in immuno_only_patients if p.pdl1_score is not None and p.pdl1_score < 0.5]
            percentage_low_pdl1 = (len(low_pdl1_immuno_only) / len(immuno_only_patients)) * 100
            print(
                f"{percentage_low_pdl1:.2f}% ({len(low_pdl1_immuno_only)} out of {len(immuno_only_patients)}) "
                f"of 'Immunotherapy Only' patients had a PDL1 score < 0.5."
            )
        else:
            print("No 'Immunotherapy Only' patients with PDL1 score data found to calculate percentage.")
    else:
        print("No patients with 'Immunotherapy Only' treatment found to calculate PDL1 percentage.")

    # 4. For patients that got immunotherapy and chemo, percentage of them that had their dosage changed
    #    Show the medications that got changed, and for each medication the avg change of dosage
    #    Also show their baackground illness
    print("\n--- Analysis 4: Dosage changes for 'Immunotherapy and Chemotherapy' patients ---")
    if combo_patients:
        if len(combo_patients) > 0:
            # Patients whose dosage was specifically changed (not 0, not None, not -1.0 for unquantifiable)
            dosage_quantifiably_changed_patients = [
                p
                for p in combo_patients
                if p.dosage_change is not None and p.dosage_change != 0.0 and p.dosage_change != -1.0
            ]
            percentage_dosage_changed = (len(dosage_quantifiably_changed_patients) / len(combo_patients)) * 100
            print(
                f"{percentage_dosage_changed:.2f}% ({len(dosage_quantifiably_changed_patients)} out of {len(combo_patients)}) "
                f"of 'Immunotherapy and Chemotherapy' patients had a quantifiable dosage change."
            )

            if dosage_quantifiably_changed_patients:
                med_dosage_changes = {}  # {med_type: [change1, change2, ...]}
                for p in dosage_quantifiably_changed_patients:
                    if p.chemotherapy_medication_type and p.chemotherapy_medication_type != "N/A":
                        # Handle multiple medications if comma-separated
                        meds = [med.strip() for med in p.chemotherapy_medication_type.split(',')]
                        for med_name in meds:
                            if med_name not in med_dosage_changes:
                                med_dosage_changes[med_name] = []
                            if p.dosage_change is not None:  # Should always be true due to filter
                                med_dosage_changes[med_name].append(p.dosage_change)

                if med_dosage_changes:
                    print("Average dosage change by medication (for those with quantifiable changes):")
                    for med, changes in med_dosage_changes.items():
                        if changes:
                            avg_change = sum(changes) / len(changes)
                            print(f"- {med}: {avg_change:.2f}% (based on {len(changes)} instance(s) of change)")
                        # else case for med with no changes shouldn't be hit due to prior filtering
                else:
                    print("No specific medications with quantifiable dosage changes were recorded for this group.")
            else:
                print("No patients in this group had a quantifiable dosage change.")
        else:
            print("No 'Immunotherapy and Chemotherapy' patients found to calculate dosage change percentage.")
    else:
        print("No patients with 'Immunotherapy and Chemotherapy' treatment found.")

    # 6. For patients that got only immunotherapy, show their background disease by percentage and sort by most common first
    print("\n--- Analysis 5: Background diseases for 'Immunotherapy Only' patients (by percentage) ---")
    if immuno_only_patients:
        if len(immuno_only_patients) > 0:
            all_background_illnesses_immuno_only = []
            for p in immuno_only_patients:
                all_background_illnesses_immuno_only.extend(p.background_illnesses)

            if all_background_illnesses_immuno_only:
                illness_counts = Counter(all_background_illnesses_immuno_only)
                total_immuno_only_patients = len(immuno_only_patients)
                print(
                    f"Background disease prevalence among {total_immuno_only_patients} 'Immunotherapy Only' patients:"
                )
                sorted_illnesses = sorted(illness_counts.items(), key=lambda item: item[1], reverse=True)
                for illness, count in sorted_illnesses:
                    percentage = (count / total_immuno_only_patients) * 100
                    print(f"- {illness}: {percentage:.2f}% ({count} patients)")
            else:
                print("No background illnesses recorded for 'Immunotherapy Only' patients.")
        else:
            print("No 'Immunotherapy Only' patients found for background disease analysis.")
    else:
        print("No patients with 'Immunotherapy Only' treatment found for background disease analysis.")
    # 7. Reasons for getting chemotherapy for patients with high PDL1, sorted by most common first
    print("\n--- Analysis 7: Reasons for getting Chemo for patients with high PDL1 (Most common first) ---")
     if combo_patients:
        if len(combo_patients) > 0:
            high_pdl1_chemo = [p for p in combo_patients if p.pdl1_score is not None and p.pdl1_score > 0.5]
            reasons_chemo = [p.reason_for_treatment for p in high_pdl1_chemo if p.reason_for_treatment]
            if reasons_chemo:
                reason_counts = Counter(reasons_chemo)
                print("Reasons:")
                for reason, count in reason_counts.most_common():
                    print(f"- \"{reason}\": {count} occurrences")
            else:
                print("No reasons specified for 'Chemo high pdl1' patients.")
    else:
        print("No patients with 'Chemoy Only' treatment found.")
    # 8. Reasons for getting immunotherapy for patients with low PDL1, sorted by most common first
    print("\n--- Analysis 7: Reasons for getting Immuno for patients with Low PDL1 (Most common first) ---")
     if immuno_only_patients:
        if len(immuno_only_patients) > 0:
            low_pdl1_immuno = [p for p in combo_patients if p.pdl1_score is not None and p.pdl1_score < 0.5]
            reasons_immuno = [p.reason_for_treatment for p in low_pdl1_immuno if p.reason_for_treatment]
            if reasons_immuno:
                reason_counts = Counter(reasons_immuno)
                print("Reasons:")
                for reason, count in reason_counts.most_common():
                    print(f"- \"{reason}\": {count} occurrences")
            else:
                print("No reasons specified for 'Immuno Only Low pdl1' patients.")
    else:
        print("No patients with 'Immuno Only' treatment found.")
    print("\n--- End of Analyses ---")


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
                    # Extract the JSON part from the LLM response,
                    # in case there's introductory/explanatory text around it.
                    json_match = re.search(r"(\{[\s\S]*\})", llm_response_text)
                    if not json_match:
                        raise json.JSONDecodeError("No JSON object found in LLM response", llm_response_text, 0)
                    json_to_parse = json_match.group(1)
                    extracted_data = json.loads(json_to_parse)

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
                except json.JSONDecodeError as e:
                    print(
                        f"ERROR: Could not parse JSON response from LLM for record {i+1}. Error: {e}. Response: {llm_response_text}"
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

        perform_analysis_and_print_results(patients_list)

except FileNotFoundError:
    print(f"ERROR: Input CSV file '{input_csv_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

print("\n\n--- Script Finished ---")
