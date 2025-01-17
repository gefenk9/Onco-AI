import os

import anthropic

API_KEY = "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"

client = anthropic.Anthropic(api_key=API_KEY)

# Define the constant system prompt
SYSTEM_PROMPT_BASE = (
    "כרופא אונקולוג, אני זקוק לסיוע בגיבוש תוכנית טיפול מקיפה עבור מטופלים שאובחנו לאחרונה עם סרטן. "
    "אתה הולך לקבל מידע על המטופל. אנא ספק מתווה מפורט של אפשרויות טיפול פוטנציאליות, "
    "כולל משטרי כימותרפיה, גישות כירורגיות, שיקולי טיפול בקרינה, וטיפולים ממוקדים על בסיס "
    "הנחיות אונקולוגיות עדכניות.בנוסף הצע בדיקות דם מתאימות לאבחנה, בנוסף, הצע אסטרטגיות לניהול תופעות לוואי נפוצות ותאר נקודות מפתח "
    "לחינוך המטופלת בנוגע לפרוגנוזה ושינויים באורח החיים. ענה בעברית בלבד. "
    "לכל המלצה הסבר את הסיבה להמלצה"
    "להלן NCCN Guidelines לפיהן עליך לענות:"
)

guidelines_content_colon_patient =''

for filename in os.listdir('NCCN_Guidlines/'):
    if filename.endswith('.txt'):
        file_path = os.path.join('NCCN_Guidlines', filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            guidelines_content_colon_patient += file.read() + '\n\n'

# Concatenate the base prompt with the guidelines content
SYSTEM_PROMPT = f"{SYSTEM_PROMPT_BASE}\n\n{guidelines_content_colon_patient}\n\n"

with open('./user_prompt.txt', 'r', encoding='utf-8') as file:
    user_prompt = file.read()
print ("\n\nLLM+SYSTEM\n\n")
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    temperature=0,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
)

print(message.content[0].text)

# print ("\n\nLLM+SYSTEM+RAG\n\n")
#
# message_rag = client.messages.create(
#     model="claude-3-5-sonnet-20241022",
#     max_tokens=1000,
#     temperature=0,
#     system="RAG_PROMPT",
#     messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
# )
#
# print(message_rag.content[0].text)

