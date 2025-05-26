import boto3
import json
import os
import re
import sys
import csv

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

# Read descriptions of all guideline files
try:
    with open('guidelines_descriptions.json', 'r', encoding='utf-8') as f:
        guidelines_preview = json.load(f)
except FileNotFoundError:
    print("ERROR: 'guidelines_descriptions.json' not found. Please ensure the file exists.")
    sys.exit(1)
except json.JSONDecodeError:
    print("ERROR: Could not decode 'guidelines_descriptions.json'. Please ensure it's valid JSON.")
    sys.exit(1)

recommended_file = './all-patient.txt'  # Ensure this fallback file exists in NCCN_Guidlines
guideline_file_path = os.path.join('NCCN_Guidlines', recommended_file)

try:
    with open(guideline_file_path, 'r', encoding='utf-8') as file:
        guidelines_content = re.sub(r' +', ' ', file.read().replace('\n', ' '))
    print(f"INFO: Successfully loaded guidelines from '{recommended_file}'.")
except Exception as e:
    print(f"ERROR reading guideline file '{guideline_file_path}': {e}")
    sys.exit(1)

# --- Define the constant system prompt for treatment plan (RAG) ---
SYSTEM_PROMPT_BASE_HE = (
    "כרופא אונקולוג, אני זקוק לסיוע בגיבוש תוכנית טיפול מקיפה עבור מטופלים שאובחנו לאחרונה עם סרטן. "
    "אתה הולך לקבל מידע על המטופל. אנא ספק מתווה מפורט של אפשרויות טיפול פוטנציאליות, "
    "כולל משטרי כימותרפיה, גישות כירורגיות, שיקולי טיפול בקרינה, וטיפולים ממוקדים על בסיס "
    "הנחיות אונקולוגיות עדכניות. בנוסף הצע בדיקות דם מתאימות לאבחנה (תפרט בבקשה את הבדיקות באופן ספציפי), בנוסף, הצע אסטרטגיות לניהול תופעות לוואי נפוצות ותאר נקודות מפתח "
    "לחינוך המטופלת בנוגע לפרוגנוזה ושינויים באורח החיים. ענה בעברית בלבד. "
    "לכל המלצה הסבר את הסיבה להמלצה. "  # Added a space here
    "להלן NCCN & ESMO Guidelines לפיהן עליך לענות:"
)

# Concatenate the base prompt with the guidelines content
system_prompt_with_guidelines = f"{SYSTEM_PROMPT_BASE_HE}\n\n{guidelines_content}\n\n"

# --- Define the constant system prompt for AI vs Doctor comparison ---
SYSTEM_PROMPT_COMPARISON_HE = (
    "אתה עוזר AI שתפקידך הוא להשוות בין המלצת טיפול שנוצרה על ידי מודל שפה גדול (LLM) לבין סיכום, מסקנות והמלצות שניתנו על ידי רופא אנושי. "
    "אנא ספק ניתוח השוואתי מפורט. התמקד בנקודות הבאות:\n"
    "1.  **דמיון**: מהן נקודות הדמיון העיקריות בין המלצת ה-LLM לבין המלצות הרופא?\n"
    "2.  **הבדלים**: מהם ההבדלים המרכזיים? האם ה-LLM הציע משהו שהרופא לא, או להיפך?\n"
    "3.  **שלמות**: האם המלצת ה-LLM מקיפה כמו זו של הרופא? האם חסרים בה אלמנטים קריטיים?\n"
    "4.  **דיוק קליני**: בהתחשב במידע המוגבל, האם המלצת ה-LLM נראית סבירה מבחינה קלינית בהשוואה לרופא? (ציין שזו הערכה ראשונית).\n"
    "5.  **הערות נוספות**: כל תובנה או הערה רלוונטית אחרת שעולה מההשוואה.\n"
    "ענה בעברית בלבד, בצורה ברורה ומובנית."
)


input_csv_path = './cases.csv'
output_csv_path = 'cases_with_ai_analysis.csv'

ORIGINAL_FIELDNAMES_HE = ['current_disease', 'summery_conclusion', 'recommendations']
NEW_FIELDNAMES_AI = ['ai_summery_conclusion', 'ai_vs_doctor_comparision']
output_fieldnames = ORIGINAL_FIELDNAMES_HE + NEW_FIELDNAMES_AI

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
            print(f"\n\n--- Processing record {i+1} from CSV ---")

            current_disease_text = row.get(ORIGINAL_FIELDNAMES_HE[0], "")
            doctor_summary_text = row.get(ORIGINAL_FIELDNAMES_HE[1], "")
            doctor_recommendations_text = row.get(ORIGINAL_FIELDNAMES_HE[2], "")

            ai_summary_conclusion = "Error: RAG call failed or no content."
            ai_vs_doctor_comparison = "Error: Comparison call failed or no content."

            # 1. First Bedrock Call: Get AI summary/conclusion (RAG)
            print("--- Invoking Bedrock (LLM+RAG) for treatment plan ---")
            rag_request_body = {
                "system": system_prompt_with_guidelines,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": [{"type": "text", "text": current_disease_text}]}],
            }
            try:
                response_rag = bedrock_client.invoke_model(
                    modelId=claude_sonnet_3_5_model_id,
                    contentType='application/json',
                    accept='application/json',
                    body=json.dumps(rag_request_body),
                )
                response_rag_text = response_rag['body'].read().decode('utf-8')
                response_rag_body_json = json.loads(response_rag_text)

                if response_rag_body_json.get("content") and len(response_rag_body_json["content"]) > 0:
                    ai_summary_conclusion = response_rag_body_json['content'][0]['text']
                else:
                    print(f"ERROR: Could not extract RAG response from Bedrock: {response_rag_body_json}")
            except Exception as e:
                print(f"ERROR during Bedrock call for RAG (record {i+1}): {e}")

            # 2. Second Bedrock Call: Get AI vs Doctor comparison
            print("\n--- Invoking Bedrock for AI vs Doctor comparison ---")
            comparison_user_prompt = f"""
הנך מתבקש להשוות את שני הטקסטים הבאים:

טקסט 1: המלצת טיפול שנוצרה על ידי LLM:
---
{ai_summary_conclusion}
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
                "max_tokens": 4000,  # Adjust if comparison needs more/less
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
                    ai_vs_doctor_comparison = response_comparison_body_json['content'][0]['text']
                else:
                    print(f"ERROR: Could not extract comparison response from Bedrock: {response_comparison_body_json}")
            except Exception as e:
                print(f"ERROR during Bedrock call for comparison (record {i+1}): {e}")

            # Write data to output CSV
            output_row = {
                ORIGINAL_FIELDNAMES_HE[0]: current_disease_text,
                ORIGINAL_FIELDNAMES_HE[1]: doctor_summary_text,
                ORIGINAL_FIELDNAMES_HE[2]: doctor_recommendations_text,
                NEW_FIELDNAMES_AI[0]: ai_summary_conclusion,
                NEW_FIELDNAMES_AI[1]: ai_vs_doctor_comparison,
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
