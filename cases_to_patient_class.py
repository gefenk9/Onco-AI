import json
import re
import sys
import csv
import time
import os
from collections import Counter  # Added for analysis
from dotenv import load_dotenv  # To read .env file for LLM_PROVIDER
from llm_client import invoke_llm  # Import the new common function

# Global file handle for output logging
output_file = None
# Save reference to original print function
_original_print = print


def tee_print(*args, **kwargs):
    """Custom print function that outputs to both stdout and file"""
    # Print to stdout as normal
    _original_print(*args, **kwargs)

    # Also write to file if it's open and not closed
    if output_file and not output_file.closed:
        # Create a copy of kwargs without 'file' parameter for file output
        file_kwargs = {k: v for k, v in kwargs.items() if k != 'file'}
        _original_print(*args, **file_kwargs, file=output_file)
        output_file.flush()  # Ensure immediate write


# Override the built-in print function
print = tee_print

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
        performance_status: int | None,
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
        self.performance_status = performance_status

    def __repr__(self):
        return (
            f"Patient(cancer_type='{self.cancer_type}', "
            f"metastasized={self.metastasized}, age={self.age}, "
            f"background_illnesses={self.background_illnesses}, "
            f"treatment_type='{self.treatment_type}', "
            f"reason_for_treatment='{self.reason_for_treatment}', "
            f"pdl1_score={self.pdl1_score}, "
            f"dosage_change={self.dosage_change}, "
            f"chemotherapy_medication_type='{self.chemotherapy_medication_type}', "
            f"performance_status={self.performance_status})"
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
    "9.  **Chemotherapy Medication Type**: (String: Applicable only if 'Treatment Type' is \"Immunotherapy and Chemotherapy\". List the type(s) of chemotherapy medication mentioned, e.g., \"R-CHOP\", \"Paclitaxel\". If multiple, join with a comma. If not applicable or not mentioned, use \"N/A\".)\n"
    "10. **Performance Status**: (Integer: ECOG Performance Status (PS) score from 0-4. If mentioned as \"PS 2\", \"ECOG 1\", \"performance status 3\", etc., extract the numeric value. If described as \"good performance status\" without a number, use null. If not mentioned, use null.)\n\n"
    "Your entire response MUST be a single, valid JSON object. Do not include any explanatory text, markdown, or any characters outside of this JSON object.\n"
    "The JSON object must have the following keys: \"cancer_type\", \"metastasized\", \"age\", \"background_illnesses\", \"treatment_type\", \"reason_for_treatment_choice\", \"pdl1_score\", \"dosage_change\", \"chemotherapy_medication_type\", \"performance_status\".\n"
    "Ensure all string fields are populated, using \"Not Specified\", \"N/A\", or \"Unknown\" where appropriate if information cannot be extracted. For lists, use an empty list if no information. For numbers/booleans, use null if not determinable."
)


def format_count_percentage(count: int, total: int, decimal_places: int = 1) -> str:
    """
    Helper function to format count and percentage in standardized format: 'X out of Y (Z%)'

    Args:
        count: The count/numerator
        total: The total/denominator
        decimal_places: Number of decimal places for percentage (default: 1)

    Returns:
        Formatted string like "5 out of 10 (50.0%)"
    """
    if total == 0:
        return f"{count} out of {total} (0.0%)"

    percentage = (count / total) * 100
    return f"{count} out of {total} ({percentage:.{decimal_places}f}%)"


def print_treatment_reasons(
    patients: list[Patient], title: str, no_patients_msg: str | None = None, no_reasons_msg: str | None = None
) -> None:
    """
    Helper function to print treatment reasons in a standardized format

    Args:
        patients: List of patients to analyze
        title: Title to print for this section
        no_patients_msg: Message to display if no patients (optional)
        no_reasons_msg: Message to display if no reasons found (optional)
    """
    if patients:
        print(f"\n{title}:")
        reasons = [p.reason_for_treatment for p in patients if p.reason_for_treatment]
        if reasons:
            reason_counts = Counter(reasons)
            for reason, count in reason_counts.most_common():
                print(f"- \"{reason}\": {count} occurrences")
        else:
            print(f"- {no_reasons_msg or 'No reasons specified'}")
    elif no_patients_msg:
        print(f"- {no_patients_msg}")


def print_background_illnesses(patients: list[Patient], title: str) -> None:
    """
    Helper function to print background illnesses in a standardized format

    Args:
        patients: List of patients to analyze
        title: Title to print for this section
    """
    print(f"\n{title}:")
    all_illnesses = []
    for p in patients:
        all_illnesses.extend(p.background_illnesses)

    if all_illnesses:
        illness_counts = Counter(all_illnesses)
        for illness, count in illness_counts.most_common():
            print(f"- \"{illness}\": {count} occurrences")
    else:
        print(f"- No background illnesses recorded")


def print_treatment_breakdown(patients: list[Patient], total_patients: int, group_name: str) -> None:
    """
    Helper function to print treatment type breakdown with percentages

    Args:
        patients: List of patients in this group
        total_patients: Total number of patients for percentage calculation
        group_name: Name of the group being analyzed
    """
    immuno_patients = [p for p in patients if p.treatment_type == "Immunotherapy Only"]
    combo_patients = [p for p in patients if p.treatment_type == "Immunotherapy and Chemotherapy"]

    if patients:
        print(f"- Immunotherapy Only: {len(immuno_patients)} ({len(immuno_patients)*100/len(patients):.1f}%)")
        print(f"- Immunotherapy and Chemotherapy: {len(combo_patients)} ({len(combo_patients)*100/len(patients):.1f}%)")
    else:
        print(f"- No {group_name} patients found")


def perform_analysis_and_print_results(patients: list[Patient]):
    print("\n\n--- Performing Analyses ---")

    if not patients:
        print("No patients to analyze.")
        return

    # Filter patients
    immuno_only_patients = [p for p in patients if p.treatment_type == "Immunotherapy Only"]
    combo_patients = [p for p in patients if p.treatment_type == "Immunotherapy and Chemotherapy"]

    print(f"Total Patients: {len(patients)}")
    print(f"Immuno Only Patients: {len(immuno_only_patients)} ({len(immuno_only_patients)*100/len(patients):.1f}%)")
    print(f"Combo Patients: {len(combo_patients)} ({len(combo_patients)*100/len(patients):.1f}%)")

    # 1. Average age of patients that only got immunotherapy
    print("\n--- Analysis 1: Average age of patients who received Immunotherapy Only ---")
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

    # 3. Reasons for getting combo, sorted by most common first
    print("\n--- Analysis 3: Reasons for getting Combo (Most common first) ---")
    if combo_patients:
        reasons_combo = [p.reason_for_treatment for p in combo_patients if p.reason_for_treatment]
        if reasons_combo:
            reason_counts = Counter(reasons_combo)
            print("Reasons:")
            for reason, count in reason_counts.most_common():
                print(f"- \"{reason}\": {count} occurrences")
        else:
            print("No reasons specified for 'Combo' patients.")
    else:
        print("No patients with 'Combo' treatment found.")

    # 4. For patients with PDL1 >= 0.5, breakdown by treatment type
    print("\n--- Analysis 4: Patients with PDL1 >= 0.5 - Treatment breakdown ---")
    high_pdl1_patients = [p for p in patients if p.pdl1_score is not None and p.pdl1_score >= 0.5]
    if high_pdl1_patients:
        high_pdl1_immuno = [p for p in high_pdl1_patients if p.treatment_type == "Immunotherapy Only"]
        high_pdl1_combo = [p for p in high_pdl1_patients if p.treatment_type == "Immunotherapy and Chemotherapy"]

        print(f"Total patients with PDL1 >= 0.5: {len(high_pdl1_patients)}")
        print(
            f"- Immunotherapy Only: {len(high_pdl1_immuno)} ({len(high_pdl1_immuno)*100/len(high_pdl1_patients):.1f}%)"
        )
        print(
            f"- Immunotherapy and Chemotherapy: {len(high_pdl1_combo)} ({len(high_pdl1_combo)*100/len(high_pdl1_patients):.1f}%)"
        )

        # Show reasons for treatment in high PDL1 patients
        print_treatment_reasons(
            high_pdl1_immuno,
            "Reasons for Immunotherapy Only (PDL1 >= 0.5)",
            no_reasons_msg="No reasons specified for high PDL1 immunotherapy only patients",
        )
        print_treatment_reasons(
            high_pdl1_combo,
            "Reasons for Immunotherapy and Chemotherapy (PDL1 >= 0.5)",
            no_reasons_msg="No reasons specified for high PDL1 combo therapy patients",
        )

        # Background illnesses for high PDL1 patients
        if high_pdl1_immuno:
            print_background_illnesses(high_pdl1_immuno, "Background illnesses for Immunotherapy Only (PDL1 >= 0.5)")
        if high_pdl1_combo:
            print_background_illnesses(
                high_pdl1_combo, "Background illnesses for Immunotherapy and Chemotherapy (PDL1 >= 0.5)"
            )
    else:
        print("No patients with PDL1 >= 0.5 found.")

    # 5. Percentage of patients that got only immunotherapy and their PDL1 is lower than 0.5
    print("\n--- Analysis 5: Percentage of 'Immunotherapy Only' patients with PDL1 < 0.5 ---")
    if immuno_only_patients:
        # Ensure there are patients to avoid division by zero if the list is empty after filtering
        if len(immuno_only_patients) > 0:
            # Get immunotherapy only patients with PDL1 data
            immuno_only_with_pdl1 = [p for p in immuno_only_patients if p.pdl1_score is not None]
            low_pdl1_immuno_only = [p for p in immuno_only_with_pdl1 if p.pdl1_score is not None and p.pdl1_score < 0.5]

            if immuno_only_with_pdl1:
                print(
                    f"{format_count_percentage(len(low_pdl1_immuno_only), len(immuno_only_with_pdl1), 2)} "
                    f"of 'Immunotherapy Only' patients with PDL1 data had a PDL1 score < 0.5."
                )

                # Performance Status >= 2 analysis for PDL1 < 0.5 patients
                if low_pdl1_immuno_only:
                    high_ps_low_pdl1 = [
                        p
                        for p in low_pdl1_immuno_only
                        if p.performance_status is not None and p.performance_status >= 2
                    ]
                    if high_ps_low_pdl1:
                        print(
                            f"- Of these PDL1 < 0.5 patients: {format_count_percentage(len(high_ps_low_pdl1), len(low_pdl1_immuno_only))} have PS >= 2"
                        )
                    else:
                        print("- No patients with PS >= 2 among PDL1 < 0.5 immunotherapy only patients")

                # Background illnesses breakdown for PDL1 < 0.5 immunotherapy only patients
                print_background_illnesses(
                    low_pdl1_immuno_only, "Background illnesses for PDL1 < 0.5 Immunotherapy Only patients"
                )
            else:
                print("No 'Immunotherapy Only' patients with PDL1 score data found.")
        else:
            print("No 'Immunotherapy Only' patients with PDL1 score data found to calculate percentage.")
    else:
        print("No patients with 'Immunotherapy Only' treatment found to calculate PDL1 percentage.")

    # 6. Percentage of all patients with PDL1 < 0.5 and treatment breakdown
    print("\n--- Analysis 6: All patients with PDL1 < 0.5 - Treatment breakdown ---")
    all_patients_with_pdl1 = [p for p in patients if p.pdl1_score is not None]
    if all_patients_with_pdl1:
        low_pdl1_all_patients = [p for p in all_patients_with_pdl1 if p.pdl1_score is not None and p.pdl1_score < 0.5]

        if low_pdl1_all_patients:
            print(
                f"Patients with PDL1 < 0.5: {format_count_percentage(len(low_pdl1_all_patients), len(all_patients_with_pdl1))} patients with PDL1 data"
            )

            # Breakdown by treatment type
            low_pdl1_immuno = [p for p in low_pdl1_all_patients if p.treatment_type == "Immunotherapy Only"]
            low_pdl1_combo = [p for p in low_pdl1_all_patients if p.treatment_type == "Immunotherapy and Chemotherapy"]

            print(
                f"- Immunotherapy Only: {len(low_pdl1_immuno)} ({len(low_pdl1_immuno)*100/len(low_pdl1_all_patients):.1f}%)"
            )
            print(
                f"- Immunotherapy and Chemotherapy: {len(low_pdl1_combo)} ({len(low_pdl1_combo)*100/len(low_pdl1_all_patients):.1f}%)"
            )

            # Reasons for treatment in low PDL1 patients
            print_treatment_reasons(
                low_pdl1_immuno,
                "Reasons for Immunotherapy Only (PDL1 < 0.5)",
                no_reasons_msg="No reasons specified for low PDL1 immunotherapy only patients",
            )
            print_treatment_reasons(
                low_pdl1_combo,
                "Reasons for Immunotherapy and Chemotherapy (PDL1 < 0.5)",
                no_reasons_msg="No reasons specified for low PDL1 combo therapy patients",
            )
        else:
            print("No patients with PDL1 < 0.5 found.")
    else:
        print("No patients with PDL1 data found.")

    # 7. Percentage of all patients with PDL1 < 0.01 and treatment breakdown
    print("\n--- Analysis 7: All patients with PDL1 < 0.01 - Treatment breakdown ---")
    if all_patients_with_pdl1:
        very_low_pdl1_all_patients = [
            p for p in all_patients_with_pdl1 if p.pdl1_score is not None and p.pdl1_score < 0.01
        ]

        if very_low_pdl1_all_patients:
            print(
                f"Patients with PDL1 < 0.01: {format_count_percentage(len(very_low_pdl1_all_patients), len(all_patients_with_pdl1))} patients with PDL1 data"
            )

            # Breakdown by treatment type
            very_low_pdl1_immuno = [p for p in very_low_pdl1_all_patients if p.treatment_type == "Immunotherapy Only"]
            very_low_pdl1_combo = [
                p for p in very_low_pdl1_all_patients if p.treatment_type == "Immunotherapy and Chemotherapy"
            ]

            print(
                f"- Immunotherapy Only: {len(very_low_pdl1_immuno)} ({len(very_low_pdl1_immuno)*100/len(very_low_pdl1_all_patients):.1f}%)"
            )
            print(
                f"- Immunotherapy and Chemotherapy: {len(very_low_pdl1_combo)} ({len(very_low_pdl1_combo)*100/len(very_low_pdl1_all_patients):.1f}%)"
            )

            # Reasons for treatment in very low PDL1 patients
            print_treatment_reasons(
                very_low_pdl1_immuno,
                "Reasons for Immunotherapy Only (PDL1 < 0.01)",
                no_reasons_msg="No reasons specified for very low PDL1 immunotherapy only patients",
            )
            print_treatment_reasons(
                very_low_pdl1_combo,
                "Reasons for Immunotherapy and Chemotherapy (PDL1 < 0.01)",
                no_reasons_msg="No reasons specified for very low PDL1 combo therapy patients",
            )
        else:
            print("No patients with PDL1 < 0.01 found.")
    else:
        print("No patients with PDL1 data found.")

    # 8. For patients that got immunotherapy and chemo, percentage of them that had their dosage changed
    #    Show the medications that got changed, and for each medication the avg change of dosage
    #    Also show their baackground illness
    print("\n--- Analysis 8: Dosage changes for 'Immunotherapy and Chemotherapy' patients ---")
    if combo_patients:
        if len(combo_patients) > 0:
            print(f"Total 'Immunotherapy and Chemotherapy' patients: {len(combo_patients)}")

            # All patients with any dosage change info (including unquantifiable)
            dosage_changed_all = [p for p in combo_patients if p.dosage_change is not None and p.dosage_change != 0.0]

            # Patients whose dosage was specifically changed (not 0, not None, not -1.0 for unquantifiable)
            dosage_quantifiably_changed_patients = [
                p
                for p in combo_patients
                if p.dosage_change is not None and p.dosage_change != 0.0 and p.dosage_change != -1.0
            ]

            # Patients with unquantifiable changes
            dosage_unquantifiable_changed = [
                p for p in combo_patients if p.dosage_change is not None and p.dosage_change == -1.0
            ]

            print(
                f"Patients with any dosage change: {format_count_percentage(len(dosage_changed_all), len(combo_patients))}"
            )
            print(
                f"- Quantifiable dosage changes: {len(dosage_quantifiably_changed_patients)} ({len(dosage_quantifiably_changed_patients)*100/len(combo_patients):.1f}%)"
            )
            print(
                f"- Unquantifiable dosage changes: {len(dosage_unquantifiable_changed)} ({len(dosage_unquantifiable_changed)*100/len(combo_patients):.1f}%)"
            )

            # Age-based analysis for dosage changes
            if dosage_changed_all:
                elderly_dosage_changed = [p for p in dosage_changed_all if p.age is not None and p.age >= 75]
                younger_dosage_changed = [p for p in dosage_changed_all if p.age is not None and p.age < 75]

                print(f"\nAge-based dosage change analysis:")
                print(
                    f"- Age >= 75: {format_count_percentage(len(elderly_dosage_changed), len(dosage_changed_all))} patients who had dosage changes"
                )
                print(
                    f"- Age < 75: {format_count_percentage(len(younger_dosage_changed), len(dosage_changed_all))} patients who had dosage changes"
                )

                # Reasons for treatment for younger patients with dosage changes
                if younger_dosage_changed:
                    print(f"\nReasons for treatment (Age < 75 with dosage changes):")
                    reasons_younger_dosage = [
                        p.reason_for_treatment for p in younger_dosage_changed if p.reason_for_treatment
                    ]
                    if reasons_younger_dosage:
                        reason_counts = Counter(reasons_younger_dosage)
                        for reason, count in reason_counts.most_common():
                            print(f"- \"{reason}\": {count} occurrences")
                    else:
                        print("- No reasons specified for younger patients with dosage changes")

            if dosage_quantifiably_changed_patients:
                med_dosage_changes = {}  # {med_type: [change1, change2, ...]}
                for p in dosage_quantifiably_changed_patients:
                    if p.chemotherapy_medication_type and p.chemotherapy_medication_type != "N/A":
                        # Handle multiple medications if comma-separated
                        for med_name in [med.strip() for med in p.chemotherapy_medication_type.split(',')]:
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

    # 9. For patients that got only immunotherapy, show their background disease by percentage and sort by most common first
    print("\n--- Analysis 9: Background diseases for 'Immunotherapy Only' patients (by percentage) ---")
    if immuno_only_patients:
        if len(immuno_only_patients) > 0:
            all_background_illnesses_immuno_only = []
            for p in immuno_only_patients:
                all_background_illnesses_immuno_only.extend(p.background_illnesses)

            if all_background_illnesses_immuno_only:
                illness_counts = Counter(all_background_illnesses_immuno_only)
                print(f"Background disease prevalence among {len(immuno_only_patients)} 'Immunotherapy Only' patients:")
                for illness, count in sorted(illness_counts.items(), key=lambda item: item[1], reverse=True):
                    percentage = (count / len(immuno_only_patients)) * 100
                    print(f"- {illness}: {percentage:.2f}% ({count} patients)")
            else:
                print("No background illnesses recorded for 'Immunotherapy Only' patients.")
        else:
            print("No 'Immunotherapy Only' patients found for background disease analysis.")
    else:
        print("No patients with 'Immunotherapy Only' treatment found for background disease analysis.")
    # 10. Reasons for getting chemotherapy for patients with high PDL1, sorted by most common first
    print("\n--- Analysis 10: Reasons for getting Chemo for patients with high PDL1 (Most common first) ---")
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
    # 11. Reasons for getting immunotherapy for patients with low PDL1, sorted by most common first
    print("\n--- Analysis 11: Reasons for getting Immuno for patients with Low PDL1 (Most common first) ---")
    if immuno_only_patients:
        if len(immuno_only_patients) > 0:
            low_pdl1_immuno = [p for p in immuno_only_patients if p.pdl1_score is not None and p.pdl1_score < 0.5]
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

    # 12. Reasons for getting chemo and immuno sorted by most common first
    print("\n--- Analysis 12: Reasons for getting chemo and immuno (Most common first) ---")
    if combo_patients:
        if len(combo_patients) > 0:
            reasons_chemo = [p.reason_for_treatment for p in combo_patients if p.reason_for_treatment]
            if reasons_chemo:
                reason_counts = Counter(reasons_chemo)
                print("Reasons:")
                for reason, count in reason_counts.most_common():
                    print(f"- \"{reason}\": {count} occurrences")
            else:
                print("No reasons specified for 'immuno and chemo' patients.")
    else:
        print("No patients with 'immuno and chemo' treatment found.")

    # 13. Performance Status >= 2 analysis
    print("\n--- Analysis 13: Patients with Performance Status >= 2 ---")
    patients_with_ps = [p for p in patients if p.performance_status is not None]
    if patients_with_ps:
        high_ps_patients = [
            p for p in patients_with_ps if p.performance_status is not None and p.performance_status >= 2
        ]

        if high_ps_patients:
            print(
                f"Patients with PS >= 2: {format_count_percentage(len(high_ps_patients), len(patients))} total patients"
            )

            # Breakdown by treatment type
            print_treatment_breakdown(high_ps_patients, len(patients), "PS >= 2")
        else:
            print("No patients with PS >= 2 found.")
    else:
        print("No patients with Performance Status data found.")

    # 14. Performance Status detailed breakdown analysis
    print("\n--- Analysis 14: Performance Status detailed breakdown ---")
    if patients_with_ps:
        # PS 0-1 group
        ps_0_1_patients = [
            p for p in patients_with_ps if p.performance_status is not None and p.performance_status <= 1
        ]
        # PS 2 group
        ps_2_patients = [p for p in patients_with_ps if p.performance_status is not None and p.performance_status == 2]
        # PS 3-4 group
        ps_3_4_patients = [
            p for p in patients_with_ps if p.performance_status is not None and p.performance_status >= 3
        ]

        print(f"Total patients with PS data: {len(patients_with_ps)}")

        # PS 0-1 analysis
        if ps_0_1_patients:
            print(f"\nPS 0-1: {format_count_percentage(len(ps_0_1_patients), len(patients))} total patients")

            print_treatment_breakdown(ps_0_1_patients, len(patients), "PS 0-1")
        else:
            print(f"\nPS 0-1: 0 out of {len(patients)} (0.0%) total patients")

        # PS 2 analysis
        if ps_2_patients:
            print(f"\nPS 2: {format_count_percentage(len(ps_2_patients), len(patients))} total patients")

            print_treatment_breakdown(ps_2_patients, len(patients), "PS 2")
        else:
            print(f"\nPS 2: 0 out of {len(patients)} (0.0%) total patients")

        # PS 3-4 analysis
        if ps_3_4_patients:
            print(f"\nPS 3-4: {format_count_percentage(len(ps_3_4_patients), len(patients))} total patients")

            print_treatment_breakdown(ps_3_4_patients, len(patients), "PS 3-4")
        else:
            print(f"\nPS 3-4: 0 out of {len(patients)} (0.0%) total patients")
    else:
        print("No patients with Performance Status data found.")

    print("\n--- End of Analyses ---")


input_csv_path = './cases.csv'
output_log_path = './cases_to_patient_class_output.txt'
ORIGINAL_FIELDNAMES = ['Current_Disease', 'Summary_Conclusions', 'Recommendations']

# Open output file for logging
try:
    output_file = open(output_log_path, 'w', encoding='utf-8')
    print(f"--- Output will be saved to: {output_log_path} ---")
except Exception as e:
    print(f"WARNING: Could not open output file {output_log_path}: {e}")
    output_file = None

try:
    with open(input_csv_path, 'r', encoding='utf-8') as infile:
        patients_list = []
        reader = csv.DictReader(infile)
        # Check that all required fields are present
        if reader.fieldnames is None:
            print(f"ERROR: Could not read CSV headers from '{input_csv_path}'.")
            sys.exit(1)

        missing_fields = [field for field in ORIGINAL_FIELDNAMES if field not in reader.fieldnames]
        if missing_fields:
            print(f"ERROR: CSV headers in '{input_csv_path}' are missing required fields: {missing_fields}.")
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
                    performance_status=None,
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
                        _original_print(
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
                        performance_status=extracted_data.get("performance_status"),  # None if missing
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
                        performance_status=None,
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
finally:
    # Close output file if it was opened
    if output_file:
        output_file.close()

print("\n\n--- Script Finished ---")
