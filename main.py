import anthropic
import os
import re
import sys

API_KEY = "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"

client = anthropic.Anthropic(api_key=API_KEY)

# Read all guideline files (first 300 lines)
guidelines_preview = {}
for filename in os.listdir('NCCN_Guidlines/'):
    if filename.endswith('.txt'):
        with open(os.path.join('NCCN_Guidlines', filename), 'r', encoding='utf-8') as file:
            lines = file.readlines()[:300]
            guidelines_preview[filename] = re.sub(r' +', ' ', ' '.join(lines))

# Read user prompt
with open('./user_prompt.txt', 'r', encoding='utf-8') as file:
    user_prompt = file.read()

# Create a system prompt to identify the relevant guideline
file_selection_prompt = (
    "You are an expert oncologist. Based on the patient description I will provide, "
    "determine which NCCN guideline file would be most appropriate to use. "
    "Here are previews of all available guideline files. Each file is separated by "
    "======= borders and clearly labeled with 'GUIDELINE FILE: filename.txt':\n\n"
)

for filename, content in guidelines_preview.items():
    file_selection_prompt += f"\n{'='*80}\n" f"GUIDELINE FILE: {filename}\n" f"{'='*80}\n" f"{content}\n" f"{'='*80}\n"

file_selection_prompt += (
    "\nBased on the following patient description, respond ONLY with exact filename list "
    "(including .txt extension) that best matches the patient's condition. "
    "Provide no other text in your response."
)

# Get the recommended guideline file
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=20,
    temperature=0,
    system=file_selection_prompt,
    messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
)

recommended_file = message.content[0].text.strip()
print(f"INFO: Recommended file '{recommended_file}'.")

# Verify file exists and read appropriate guideline
try:
    if not os.path.exists(os.path.join('NCCN_Guidlines', recommended_file)):
        print(f"WARNING: Recommended file '{recommended_file}' not found. Using all-patient.txt as fallback.")
        recommended_file = 'all-patient.txt'

    with open(os.path.join('NCCN_Guidlines', recommended_file), 'r', encoding='utf-8') as file:
        guidelines_content = re.sub(r' +', ' ', file.read().replace('\n', ' '))

except Exception as e:
    print(f"ERROR reading guideline file: {e}")
    sys.exit(1)

# Define the constant system prompt
SYSTEM_PROMPT_BASE = (
    "כרופא אונקולוג, אני זקוק לסיוע בגיבוש תוכנית טיפול מקיפה עבור מטופלים שאובחנו לאחרונה עם סרטן. "
    "אתה הולך לקבל מידע על המטופל. אנא ספק מתווה מפורט של אפשרויות טיפול פוטנציאליות, "
    "כולל משטרי כימותרפיה, גישות כירורגיות, שיקולי טיפול בקרינה, וטיפולים ממוקדים על בסיס "
    "הנחיות אונקולוגיות עדכניות.בנוסף הצע בדיקות דם מתאימות לאבחנה(תפרט בבקשה את הבדיקות באופן ספציפי), בנוסף, הצע אסטרטגיות לניהול תופעות לוואי נפוצות ותאר נקודות מפתח "
    "לחינוך המטופלת בנוגע לפרוגנוזה ושינויים באורח החיים. ענה בעברית בלבד. "
    "לכל המלצה הסבר את הסיבה להמלצה"
    "להלן NCCN Guidelines וESMO Guidelinesלפיהן עליך לענות:"
)

# Concatenate the base prompt with the guidelines content
SYSTEM_PROMPT = f"{SYSTEM_PROMPT_BASE}\n\n{guidelines_content}\n\n"

with open('./user_prompt.txt', 'r', encoding='utf-8') as file:
    user_prompt = file.read()

print ("\n\nLLM+RAG\n\n")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2500,
    temperature=0,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
)

print(message.content[0].text)

print ("\n\nLLM\n\n")

message_rag = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    temperature=0,
    system="SYSTEM_PROMPT_BASE",
    messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
)

print(message_rag.content[0].text)
