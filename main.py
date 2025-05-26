import boto3
import json
import os  # Was missing import
import re  # Was missing import
import sys  # Was missing import

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

# Read user prompt
try:
    with open('./user_prompt.txt', 'r', encoding='utf-8') as file:
        user_prompt_text = file.read()
except FileNotFoundError:
    print("ERROR: './user_prompt.txt' not found. Please ensure the file exists.")
    sys.exit(1)

recommended_file = 'all-patient.txt'  # Ensure this fallback file exists in NCCN_Guidlines
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

print("\n\n--- Invoking Bedrock (LLM+RAG) for treatment plan ---")

rag_request_body = {
    "system": system_prompt_with_guidelines,
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 4000,  # Adjusted from 5000 as Claude 3.5 Sonnet has context window limits, be mindful
    "temperature": 0.0,  # For consistent output, adjust if creativity is needed
    "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt_text}]}],
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
        claude_response_rag = response_rag_body_json['content'][0]['text']
        print("\n--- Bedrock (LLM+RAG) Response: ---")
        print(claude_response_rag)
    else:
        print(f"ERROR: Could not extract RAG response from Bedrock: {response_rag_body_json}")

except Exception as e:
    print(f"ERROR during Bedrock call for RAG: {e}")


print("\n\n--- Script Finished ---")
