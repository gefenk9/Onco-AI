import boto3
import json
import os
import re
import sys
import csv
import time  # Import the time module

# Configs
REQUEST_DELAY_SECONDS = 31  # Delay in seconds between requests (current rate limit is 2 req/sec)

# AWS Bedrock client initialization
aws_region = 'eu-west-1'  # Your AWS region for Bedrock
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=aws_region,
    # If running on an EC2 instance with an IAM role,
    # or if your AWS credentials are configured in the environment/AWS CLI,
    # you don't need to pass aws_access_key_id and aws_secret_access_key here.
)

# Corrected Model ID for Claude 3.5 Sonnet on Bedrock
# The region prefix "eu." is not part of the modelId;
# the region is specified when creating the bedrock_client.
claude_sonnet_3_5_model_id = "eu.anthropic.claude-3-5-sonnet-20240620-v1:0"

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

ORIGINAL_FIELDNAMES_HE = ['current_disease', 'summery_conclusion', 'recommendations']
NEW_FIELDNAMES = ['llm_summery_conclusion', 'llm_vs_doctor_comparision', 'llm_vs_doctor_comparison_score']
output_fieldnames = ORIGINAL_FIELDNAMES_HE + NEW_FIELDNAMES

try:
    with open(input_csv_path, 'r', encoding='utf-8') as infile, open(
        output_csv_path, 'w', newline='', encoding='utf-8'
    ) as outfile:

        reader = csv.DictReader(infile)
        # Ensure the reader uses the correct fieldnames if they are not exactly as expected
        if reader.fieldnames != ORIGINAL_FIELDNAMES_HE:
            print(
                f"WARNING: CSV headers in '{input_csv_path}' are {reader.fieldnames}, expected {ORIGINAL_FIELDNAMES_HE}."
            )
            # If critical, you might want to sys.exit(1) or adapt ORIGINAL_FIELDNAMES_HE

        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            if i > 0:  # If it's not the first record, wait before processing this new record
                print(f"\n--- Waiting {REQUEST_DELAY_SECONDS} seconds before processing record {i+1}... ---")
                time.sleep(REQUEST_DELAY_SECONDS)  # Wait for the configured duration

            print(f"\n\n--- Processing record {i+1} from CSV ---")

            current_disease_text = row.get(ORIGINAL_FIELDNAMES_HE[0], "")
            doctor_summary_text = row.get(ORIGINAL_FIELDNAMES_HE[1], "")
            doctor_recommendations_text = row.get(ORIGINAL_FIELDNAMES_HE[2], "")

            llm_summary_conclusion = "Error: LLM call failed or no content."
            llm_vs_doctor_comparison = "Error: Comparison call failed or no content."
            llm_comparison_score = DEFAULT_SCORE_ON_ERROR

            # 1. First Bedrock Call: Get AI summary/conclusion
            print("--- Invoking Bedrock LLM for treatment plan ---")
            llm_request_body = {
                "system": SYSTEM_PROMPT_BASE_HE,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": [{"type": "text", "text": current_disease_text}]}],
            }
            try:
                response_llm = bedrock_client.invoke_model(
                    modelId=claude_sonnet_3_5_model_id,
                    contentType='application/json',
                    accept='application/json',
                    body=json.dumps(llm_request_body),
                )
                response_llm_text = response_llm['body'].read().decode('utf-8')
                response_llm_body_json = json.loads(response_llm_text)

                if response_llm_body_json.get("content") and len(response_llm_body_json["content"]) > 0:
                    llm_summary_conclusion = response_llm_body_json['content'][0]['text']
                else:
                    print(f"ERROR: Could not extract LLM response from Bedrock: {response_llm_body_json}")
            except Exception as e:
                print(f"ERROR during Bedrock call for LLM (record {i+1}): {e}")

            # Wait before the second AI request for the current record
            print(
                f"\n--- Waiting {REQUEST_DELAY_SECONDS} seconds before AI vs Doctor comparison request for record {i+1}... ---"
            )
            time.sleep(REQUEST_DELAY_SECONDS)

            # 2. Second Bedrock Call: Get AI vs Doctor comparison
            print("\n--- Invoking Bedrock for AI vs Doctor comparison ---")
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
            comparison_request_body = {
                "system": SYSTEM_PROMPT_COMPARISON_HE,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 3000,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": [{"type": "text", "text": comparison_user_prompt}]}],
            }
            try:
                response_comparison = bedrock_client.invoke_model(
                    modelId=claude_sonnet_3_5_model_id,
                    contentType='application/json',
                    accept='application/json',
                    body=json.dumps(comparison_request_body),
                )
                response_comparison_text = response_comparison['body'].read().decode('utf-8')
                response_comparison_body_json = json.loads(response_comparison_text)

                if response_comparison_body_json.get("content") and len(response_comparison_body_json["content"]) > 0:
                    raw_comparison_text = response_comparison_body_json['content'][0]['text']

                    # Extract score and clean the text
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
                else:
                    print(f"ERROR: Could not extract comparison response from Bedrock: {response_comparison_body_json}")
                    llm_comparison_score = DEFAULT_SCORE_ON_ERROR
            except Exception as e:
                print(f"ERROR during Bedrock call for comparison (record {i+1}): {e}")
                llm_comparison_score = DEFAULT_SCORE_ON_ERROR

            # Write data to output CSV
            output_row = {
                ORIGINAL_FIELDNAMES_HE[0]: current_disease_text,
                ORIGINAL_FIELDNAMES_HE[1]: doctor_summary_text,
                ORIGINAL_FIELDNAMES_HE[2]: doctor_recommendations_text,
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
