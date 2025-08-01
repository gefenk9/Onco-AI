import json
import re
import sys
import csv
import time
import os
from llm_client import invoke_llm  # Import the new common function

# Configs
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure_openai").lower()
REQUEST_DELAY_SECONDS = 31  # Delay in seconds between requests (current rate limit is 2 req/sec)
ANTHROPIC_NO_RATE_LIMIT = LLM_PROVIDER == "anthropic"

# Define the constant system prompt for treatment plan
SYSTEM_PROMPT_BASE_HE = (
    "כרופא אונקולוג, אני זקוק לסיוע בגיבוש תוכנית טיפול מקיפה עבור מטופלים שאובחנו לאחרונה עם סרטן. "
    "אתה הולך לקבל מידע על המטופל. אנא ספק מתווה מפורט של אפשרויות טיפול פוטנציאליות, "
    "כולל משטרי כימותרפיה, גישות כירורגיות, שיקולי טיפול בקרינה, וטיפולים ממוקדים על בסיס "
    "הידע הקיים שלך על הנחיות NCCN & ESMO העדכניות ביותר. "
    "בנוסף הצע בדיקות דם מתאימות לאבחנה (תפרט בבקשה את הבדיקות באופן ספציפי), בנוסף, הצע אסטרטגיות לניהול תופעות לוואי נפוצות ותאר נקודות מפתח "
    "לחינוך המטופלת בנוגע לפרוגנוזה ושינויים באורח החיים. ענה בעברית בלבד. "
    "לכל המלצה הסבר את הסיבה להמלצה. "
)


# Define the constant system prompt for AI vs Doctor comparison
SYSTEM_PROMPT_COMPARISON_HE = (
    "אתה עוזר AI שתפקידך הוא להשוות בין המלצת טיפול שנוצרה על ידי מודל שפה גדול (LLM) לבין סיכום, מסקנות והמלצות שניתנו על ידי רופא אנושי. "
    "אנא ספק ניתוח השוואתי מפורט. התמקד בנקודות הבאות:\n"
    "1.  **דמיון**: מהן נקודות הדמיון העיקריות בין המלצת ה-LLM לבין המלצות הרופא?\n"
    "2.  **הבדלים**: מהם ההבדלים המרכזיים? האם ה-LLM הציע משהו שהרופא לא, או להיפך?\n"
    "3.  **שלמות**: האם המלצת ה-LLM מקיפה כמו זו של הרופא? האם חסרים בה אלמנטים קריטיים?\n"
    "4.  **דיוק קליני**: בהתחשב במידע המוגבל, האם המלצת ה-LLM נראית סבירה מבחינה קלינית בהשוואה לרופא? (ציין שזו הערכה ראשונית).\n"
    "5.  **הערות נוספות**: כל תובנה או הערה רלוונטית אחרת שעולה מההשוואה.\n"
    "6.  **ציון דמיון מספרי**: בסוף הניתוח שלך, אנא הוסף שורה נפרדת עם ציון הדמיון המספרי בין המלצת ה-LLM להמלצת הרופא. הציון צריך להיות בין 0 (לא דומה כלל) ל-1 (דומה מאוד). השתמש בפורמט הבא בדיוק: `ציון דמיון מספרי (0-1): [הציון שלך]` (לדוגמה: `ציון דמיון מספרי (0-1): 0.85`).\n"
    "ענה בעברית בלבד, בצורה ברורה ומובנית."
)


input_csv_path = './cases.csv'
output_csv_path = 'cases_with_analysis.csv'
DEFAULT_SCORE_ON_ERROR = 0.0
CSV_FIELD_DISEASE = 'Current_Disease'
CSV_FIELD_SUMMARY_CONCLUSION = 'Summary_Conclusions'
CSV_FIELD_RECOMMENDATIONS = 'Recommendations'
ORIGINAL_FIELDNAMES = ['PatId', 'Current_Disease', 'Summary_Conclusions', 'Recommendations']
NEW_FIELDNAMES = ['Llm_Summary_Conclusions', 'Llm_Vs_Doctor_Comparision', 'Llm_Vs_Doctor_Comparison_Score']
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

            llm_summary_conclusion = "Error: LLM call failed or no content."
            llm_vs_doctor_comparison = "Error: Comparison call failed or no content."
            llm_comparison_score = DEFAULT_SCORE_ON_ERROR

            # 1. First LLM Call: Get AI summary/conclusion
            print("--- Invoking LLM for treatment plan ---")

            llm_response_text = invoke_llm(
                system_prompt=SYSTEM_PROMPT_BASE_HE,
                user_prompt_text=current_disease_text,
                max_tokens=1000,
                temperature=0.0,
                # provider_override can be used here if needed, e.g., os.getenv("LLM_PROVIDER_CASES", "bedrock")
            )

            if llm_response_text.startswith("ERROR:"):
                print(f"ERROR during LLM call for treatment plan (record {i+1}): {llm_response_text}")
                # llm_Summary_Conclusions remains "Error: LLM call failed or no content." (its default)
            else:
                llm_summary_conclusion = llm_response_text

            # Ensure llm_summary_conclusion has a value for the next step, even if it's an error message
            if llm_summary_conclusion == "Error: LLM call failed or no content." and not llm_response_text.startswith(
                "ERROR:"
            ):
                llm_summary_conclusion = llm_response_text  # Should not happen if logic is correct

            if not ANTHROPIC_NO_RATE_LIMIT:
                # Wait before the second LLM request for the current record
                print(
                    f"\n--- Waiting {REQUEST_DELAY_SECONDS} seconds before AI vs Doctor comparison request for record {i+1}... ---"
                )
                time.sleep(REQUEST_DELAY_SECONDS)

            # 2. Second LLM Call: Get LLM vs Doctor comparison
            print("\n--- Invoking LLM vs Doctor comparison ---")
            comparison_user_prompt = f"""
הנך מתבקש להשוות את שני הטקסטים הבאים:

טקסט 1: המלצת טיפול שנוצרה על ידי LLM:
---
{llm_summary_conclusion}
---

טקסט 2: סיכום, מסקנות והמלצות של רופא אנושי:
---
סיכום ומסקנות הרופא:
{doctor_summary_text}

המלצות הרופא:
{doctor_recommendations_text}
---

אנא ספק את ניתוח ההשוואה שלך בהתאם להנחיות שקיבלת.
"""

            comparison_response_text = invoke_llm(
                system_prompt=SYSTEM_PROMPT_COMPARISON_HE,
                user_prompt_text=comparison_user_prompt,
                max_tokens=3000,
                temperature=0.0,
            )

            if comparison_response_text.startswith("ERROR:"):
                print(f"ERROR during LLM call for comparison (record {i+1}): {comparison_response_text}")
                # llm_vs_doctor_comparison remains "Error: Comparison call failed or no content."
                # llm_comparison_score remains DEFAULT_SCORE_ON_ERROR
            else:
                raw_comparison_text = comparison_response_text
                try:
                    # Extract numerical score and clean the text
                    score_extraction_pattern = r"ציון דמיון מספרי \(0-1\):\s*(0-1?)"
                    score_match = re.search(score_extraction_pattern, raw_comparison_text)

                    if score_match:
                        try:
                            llm_comparison_score = float(score_match.group(1))
                            # Remove the score line from the comparison text
                            score_line_removal_pattern = r"^\s*ציון דמיון מספרי \(0-1\):\s*0-1?\s*[\r\n]?"
                            # Using re.MULTILINE to ensure ^ matches the start of a line
                            llm_vs_doctor_comparison = re.sub(
                                score_line_removal_pattern, "", raw_comparison_text, count=1, flags=re.MULTILINE
                            ).strip()
                        except ValueError:
                            print(
                                f"WARNING: Could not convert extracted score '{score_match.group(1)}' to float for record {i+1}. Score set to {DEFAULT_SCORE_ON_ERROR}."
                            )
                            llm_comparison_score = DEFAULT_SCORE_ON_ERROR
                            llm_vs_doctor_comparison = raw_comparison_text  # Keep raw text if score parsing failed
                        except Exception as parse_ex:
                            print(
                                f"WARNING: Error processing score for record {i+1}: {parse_ex}. Score set to {DEFAULT_SCORE_ON_ERROR}."
                            )
                            llm_comparison_score = DEFAULT_SCORE_ON_ERROR
                            llm_vs_doctor_comparison = raw_comparison_text
                    else:
                        print(
                            f"WARNING: Similarity score line not found in LLM comparison response for record {i+1}. Score set to {DEFAULT_SCORE_ON_ERROR}."
                        )
                        llm_vs_doctor_comparison = raw_comparison_text  # Use raw text if score line not found
                        llm_comparison_score = DEFAULT_SCORE_ON_ERROR
                except Exception as e:  # Catch any other unexpected error during parsing
                    print(f"ERROR processing comparison response (record {i+1}): {e}")
                    llm_vs_doctor_comparison = raw_comparison_text  # Keep raw text
                    llm_comparison_score = DEFAULT_SCORE_ON_ERROR
            # Write data to output CSV
            output_row = {
                'PatId': pat_id,
                'Current_Disease': current_disease_text,
                'Summary_Conclusions': doctor_summary_text,
                'Recommendations': doctor_recommendations_text,
                NEW_FIELDNAMES[0]: llm_summary_conclusion,
                NEW_FIELDNAMES[1]: llm_vs_doctor_comparison,
                NEW_FIELDNAMES[2]: llm_comparison_score,
            }
            writer.writerow(output_row)
            print(f"--- Finished processing and wrote record {i+1} to '{output_csv_path}' ---")

except FileNotFoundError:
    print(f"ERROR: Input CSV file '{input_csv_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

print("\n\n--- Script Finished ---")
