import csv
import sys
import argparse
from llm_client import invoke_llm  # Import the new common function

# --- Configuration ---
INPUT_CSV_PATH = './cases.csv'
OUTPUT_TXT_PATH = 'cross_analysis_output.txt'
DEFAULT_MAX_RECORDS = 50
CSV_FIELD_DISEASE = 'Current_Disease'
CSV_FIELD_SUMMARY_CONCLUSION = 'Summary_Conclusions'
CSV_FIELD_RECOMMENDATIONS = 'Recommendations'
EXPECTED_CSV_HEADERS = ['Current_Disease', 'Summary_Conclusions', 'Recommendations']


# --- System Prompt for LLM ---
SYSTEM_PROMPT_CROSS_ANALYSIS_HE = (
    "כרופא אונקולוג מומחה, אתה מתבקש לבצע ניתוח-על (cross-analysis) של המלצות טיפול ותיאורי מחלה שניתנו במקרים אונקולוגיים שונים. "
    "לכל מקרה, תקבל את תיאור המחלה, סיכום ומסקנות הרופא, והמלצות הטיפול של הרופא. "
    "הניתוח על צריך להיות בדגש על ההמלצות הטיפוליות - אימונתרפיה בלבד או אימונתרפיה וכימותרפיה ביחד, אלו ששני ההמלצות היחידות, על כל מסקנה שאתה רושם התייחס גם להמלצות אלו מבחינה כמותית ואחוזים ביחד"
    "חשוב מאוד שלכל תת קבוצה וסיבתיות תציין את כמות האנשים שנכללים בפקטור הזה בשביל טיפול מסויים כמו אימונו בלבד, לכל משפט שאתה כותב תציין אחוזים"
    "המטרה היא לזהות מאפיינים משותפים, דפוסים, או קשרים בין סוגי ההמלצות הטיפוליות לבין מאפייני המקרים בצורה כמותית (כגון סוג המחלה, שלב משוער, או מאפיינים קליניים אחרים שניתן להסיק מהמידע שנמסר, כולל מסיכום הרופא). "
    "תחפש לי גורמים מנבאים בטקסט - אם הרופא מנמק את ההחלטה אם היא חריגה רק טיפול אימוני או שאינו מנמק את ההחלטה אם היא רגילה שזה אומר שהוא משלב כימי אימוני סטנדרטי "
    "בסס את הניתוח שלך על הידע הקיים שלך בהנחיות NCCN ו-ESMO העדכניות ביותר. "
    "אנא הצג את הממצאים שלך בצורה כמותית של אחוזים על פני כלל המטופלים, רצוי לקבץ אותם לפי סוג ההמלצה העיקרי, לפי סוג המחלה, או לפי מאפיינים משותפים משמעותיים שזיהית. התייחס גם לסיכום ומסקנות הרופא כחלק מהקונטקסט לניתוח ההמלצות. "
    ", בנוסף אתה מתבקש לענות על השאלות הבאות:איזה מחלות רקע הובילו להפחתת מינון , באיזה תרופות, כמה הפחתת מינון - תציג את הכל באחוזים"
    "מה היו הטיפולים הפחות אגרסיביים שהוצעו בחולים במצב תפקודי ירוד וכמה באחוזים"
    "מה הגיל שמעליו לרוב בוצעו הפחתות מינון או שינוי בטיפול"
    "האם חולים שקיבלו אימונותרפיה ללא כימו וpdl1 פחות מ50% היו כאלה במצב תפקודי ירוד/ מבוגרים / מחלות רקע? או הסבר אחר"
    "מדוע מטופלים עם pdl1 מעל 50% קיבלו גם כימו וגם אימונותרפיה ולא אימונו לבד?. "
    "לדוגמה, אם מספר מקרים עם המלצה לכימותרפיה חולקים מאפיין מסוים מה הוא באחוזים (למשל, שלב מחלה מתקדם המשתמע מההמלצה או תיאור המחלה), ציין זאת. "
    "חשוב לי שעל כל נתון שאתה מספר כמו אחוז שאתה מציין תציין את כמות האנשים ותסביר מתוך איזה קבוצה חישבת את האחוז הזה, כמה הקבוצה הזו מכילה ומה המכנה המשותף שלה"
    "בנוסף עבור כל מנבא מו גיל מבוגר, renal implement , PS תרשום כמה מתוכם קיבלו אימונו וכמה קיבלו כימו אימונו"
    "תתייחס לכל מי שקיבל טיפול כימוטרפי בלבד שהוא יקבל בעתיד טיפול אימונותרפי גם כן ולכן הקבוצות העיקריות שיש להתייחס אליהם בכל נתן של אחוזים וכמות מטופלים צריך לציין מאיזה קבוצה היא וכמה אנשים מתוך הקבוצה של כימו ואימונו או רק אימונו ולמה"
    "הניתוח צריך להיות מעמיק ומבוסס על הנתונים שסופקו. "
    "ענה באנגלית בלבד."
)


def prepare_llm_user_prompt(cases_data_list):
    """
    Prepares the user prompt string for the LLM, containing all case data.
    """
    intro = """להלן מספר מקרים אונקולוגיים, כל אחד עם תיאור המחלה והמלצות הרופא.
אנא נתח מקרים אלו כדי למצוא מאפיינים משותפים בין המטופלים או המחלות, בהתבסס על סוגי ההמלצות שניתנו, תיאורי המחלה, וסיכום/מסקנות הרופא.
התמקד במציאת דפוסים הקושרים סוגי המלצות (כגון כימותרפיה, כירורגיה, טיפול קרינתי, אימונותרפיה, טיפול ביולוגי, טיפול תומך/פליאטיבי) למאפייני מקרה.
ארגן את התשובה שלך בצורה ברורה ומפורטת, קבץ את הממצאים לפי סוג ההמלצה, סוג המחלה, או מאפיינים משותפים אחרים שאתה מזהה.

מקרים לניתוח:
"""
    cases_str_parts = []
    for i, case_data in enumerate(cases_data_list):
        case_str = f"""
--- מקרה {i+1} ---
תיאור המחלה:
{case_data[CSV_FIELD_DISEASE]}

סיכום ומסקנות הרופא:
{case_data[CSV_FIELD_SUMMARY_CONCLUSION]}

המלצות הרופא:
{case_data[CSV_FIELD_RECOMMENDATIONS]}
--- סוף מקרה {i+1} ---
"""
        cases_str_parts.append(case_str)

    if not cases_str_parts:
        return None  # No data to process

    return (
        intro + "\n".join(cases_str_parts) + "\n\nאנא ספק את הניתוח המבוקש, תוך התייחסות להנחיות שקיבלת בפרומפט המערכת."
    )


def main():
    parser = argparse.ArgumentParser(description="Perform cross-analysis of oncology cases using an LLM.")
    parser.add_argument(
        "--max_records",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help=f"Maximum number of records from {INPUT_CSV_PATH} to process (default: {DEFAULT_MAX_RECORDS}).",
    )
    args = parser.parse_args()
    max_records_to_process = args.max_records

    print(f"--- Starting cross-analysis script ---")
    print(f"Processing up to {max_records_to_process} records from '{INPUT_CSV_PATH}'.")

    cases_data = []
    try:
        with open(INPUT_CSV_PATH, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            # First, check if fieldnames is None (e.g., empty CSV)
            if reader.fieldnames is None:
                print(f"ERROR: CSV file '{INPUT_CSV_PATH}' appears to be empty or has no header row. Exiting.")
                sys.exit(1)
            elif reader.fieldnames != EXPECTED_CSV_HEADERS:
                print(
                    f"WARNING: CSV headers in '{INPUT_CSV_PATH}' are {reader.fieldnames}, "
                    f"expected {EXPECTED_CSV_HEADERS}. Proceeding with caution."
                )
                # Check if essential columns are present
                if not (
                    CSV_FIELD_DISEASE in reader.fieldnames
                    and CSV_FIELD_SUMMARY_CONCLUSION in reader.fieldnames
                    and CSV_FIELD_RECOMMENDATIONS in reader.fieldnames
                ):
                    print(
                        f"ERROR: Essential columns ('{CSV_FIELD_DISEASE}', '{CSV_FIELD_SUMMARY_CONCLUSION}', '{CSV_FIELD_RECOMMENDATIONS}') not found in CSV headers. Exiting."
                    )
                    sys.exit(1)

            for i, row in enumerate(reader):
                if i >= max_records_to_process:
                    print(f"Reached max_records limit of {max_records_to_process}.")
                    break

                disease_info = row.get(CSV_FIELD_DISEASE)
                recommendations_info = row.get(CSV_FIELD_RECOMMENDATIONS)
                summary_conclusion_info = row.get(CSV_FIELD_SUMMARY_CONCLUSION)

                if not disease_info or not recommendations_info or not summary_conclusion_info:
                    print(
                        f"Warning: Skipping record {i+1} due to missing {CSV_FIELD_DISEASE}, {CSV_FIELD_SUMMARY_CONCLUSION}, or {CSV_FIELD_RECOMMENDATIONS}."
                    )
                    continue

                cases_data.append(
                    {
                        CSV_FIELD_DISEASE: disease_info,
                        CSV_FIELD_SUMMARY_CONCLUSION: summary_conclusion_info,
                        CSV_FIELD_RECOMMENDATIONS: recommendations_info,
                    }
                )
            print(f"Read {len(cases_data)} records from CSV.")

    except FileNotFoundError:
        print(f"ERROR: Input CSV file '{INPUT_CSV_PATH}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        sys.exit(1)

    if not cases_data:
        print("No data processed from CSV. Exiting.")
        sys.exit(0)

    user_prompt = prepare_llm_user_prompt(cases_data)
    if not user_prompt:
        print("Failed to generate user prompt (no data). Exiting.")
        sys.exit(1)

    print("--- Invoking LLM for cross-analysis ---")
    llm_analysis_result = invoke_llm(
        system_prompt=SYSTEM_PROMPT_CROSS_ANALYSIS_HE,
        user_prompt_text=user_prompt,
        max_tokens=4000,
        temperature=0.1,
        # provider_override can be used here if needed
    )

    if llm_analysis_result.startswith("ERROR:"):
        print(f"LLM cross-analysis failed: {llm_analysis_result}")
        # Decide if to exit or write the error to file
        # Current behavior is to write the error to file
    else:
        print("--- LLM cross-analysis successful ---")

    try:
        with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as outfile:
            outfile.write(llm_analysis_result)
        print(f"--- LLM analysis successfully written to '{OUTPUT_TXT_PATH}' ---")
    except Exception as e:
        print(f"Error writing LLM analysis to file '{OUTPUT_TXT_PATH}': {e}")

    print("--- Script Finished ---")


if __name__ == "__main__":
    main()
