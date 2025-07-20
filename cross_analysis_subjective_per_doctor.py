import pandas as pd
import sys
import argparse
import os
from llm_client import invoke_llm

# --- Configuration ---
INPUT_CSV_PATH_CASES = './cases.csv'
INPUT_CSV_PATH_DOCTORS = './doctors.csv'
OUTPUT_RAW_DATA_FOLDER = './raw_data'
DEFAULT_MAX_RECORDS_PER_DOCTOR = 100
CSV_FIELD_PAT_ID = 'PatID'

# --- System Prompt for LLM (in Hebrew) ---
SYSTEM_PROMPT_CROSS_ANALYSIS_HE = (
    "כאונקולוג מומחה, משימתך היא לנתח את החלטות הטיפול של רופא ספציפי על סמך מקרי מטופלים שיוצגו בפניך. המטרה המרכזית היא לחשוף דפוסים, הטיות ורמזים נסתרים בתהליך קבלת ההחלטות של הרופא."
    "עליך לבחון לעומק כיצד משתנים כמו גיל המטופל, מינו, עיר מגוריו, ומצבו הסוציו-אקונומי (כפי שניתן להסיק מהנתונים) עשויים להשפיע על המלצות הטיפול."
    "בנוסף, נתח את הקשר בין מחלות רקע, שלב המחלה, וסוג הטיפול הנבחר או המינון שלו."
    "לדוגמה, האם הרופא נוטה להמליץ על טיפול אגרסיבי יותר למטופלים צעירים? האם ישנה העדפה לטיפולים מסוימים בערים מסוימות? האם מחלות רקע ספציפיות משפיעות באופן עקבי על החלטות הטיפול?"
    "הצג את ממצאיך בצורה מפורטת, מנומקת ומגובה בנתונים מהמקרים שסופקו. בסס את הניתוח שלך על הידע העדכני ביותר ועל קווי המנחה של NCCN ו-ESMO."
    "הניתוח שלך צריך להיות מבוסס על הנתונים שסופקו בלבד. אנא כתוב את תשובתך בעברית."
)


def prepare_llm_user_prompt(doctor_name, cases_data_df):
    """
    Prepares the user prompt string for the LLM, containing all case data for a single doctor.
    """
    prompt = f'להלן מספר מקרים אונקולוגיים שטופלו על ידי ד"ר {doctor_name}.\n'
    prompt += 'אנא נתח את המקרים הללו כדי למצוא מאפיינים משותפים, דפוסים או הטיות בקבלת ההחלטות של הרופא, בהתבסס על הנחיותיך.\n\n'
    prompt += 'מקרים לניתוח:\n'

    cases_list = cases_data_df.to_dict(orient='records')
    
    for i, case_data in enumerate(cases_list):
        prompt += f'\n--- מקרה מספר {i+1} ---\n'
        case_details = []
        for key, value in case_data.items():
            if pd.notna(value) and value != '':
                case_details.append(f'{key}: {value}')
        prompt += ", ".join(case_details)
        prompt += f'\n--- סוף מקרה מספר {i+1} ---\n'

    if len(cases_list) == 0:
        return None

    prompt += '\n\nאנא ספק את הניתוח המבוקש, בהתאם להנחיות שקיבלת.'
    return prompt


def main():
    parser = argparse.ArgumentParser(description="בצע ניתוח-על של מקרים אונקולוגיים לכל רופא באמצעות LLM.")
    parser.add_argument(
        "--max_records_per_doctor",
        type=int,
        default=DEFAULT_MAX_RECORDS_PER_DOCTOR,
        help="מספר המקרים המקסימלי לעבד לכל רופא (ברירת מחדל: {DEFAULT_MAX_RECORDS_PER_DOCTOR}).",
    )
    args = parser.parse_args()
    max_records_to_process = args.max_records_per_doctor

    print("--- מתחיל סקריפט ניתוח-על לפי רופא ---")

    # --- Load and Merge Data ---
    try:
        cases_df = pd.read_csv(INPUT_CSV_PATH_CASES, dtype={CSV_FIELD_PAT_ID: str})
        doctors_df = pd.read_csv(INPUT_CSV_PATH_DOCTORS, dtype={CSV_FIELD_PAT_ID: str})
        print(f'קריאת {len(cases_df)} מקרים ו-{len(doctors_df)} רשומות רופאים הושלמה.')

        # Merge the dataframes on the patient ID
        merged_df = pd.merge(cases_df, doctors_df, on=CSV_FIELD_PAT_ID)
        print(f'נמצאו {len(merged_df)} מקרים עם התאמה לרופאים.')

        if merged_df.empty:
            print("לא נמצאו מקרים משותפים בין הקבצים. יוצא מהתוכנית.")
            sys.exit(0)

        # Create a full name for the doctor
        merged_df['doctor_name'] = merged_df['Doc_First_Name'] + " " + merged_df['Doc_Last_Name']

    except FileNotFoundError as e:
        print(f"שגיאה: קובץ לא נמצא - {e}. ודא שהקבצים '{INPUT_CSV_PATH_CASES}' ו-'{INPUT_CSV_PATH_DOCTORS}' קיימים.")
        sys.exit(1)
    except Exception as e:
        print(f"שגיאה כללית בקריאת ומיזוג הנתונים: {e}")
        sys.exit(1)

    # Group cases by doctor
    grouped_by_doctor = merged_df.groupby('doctor_name')

    os.makedirs(OUTPUT_RAW_DATA_FOLDER, exist_ok=True)

    # --- Loop Through Each Doctor for Analysis ---
    for doctor_name, doctor_cases_df in grouped_by_doctor:
        print(f'\n--- מתחיל ניתוח עבור ד"ר {doctor_name} ---')
        
        # Limit the number of records if specified
        doctor_cases_df = doctor_cases_df.head(max_records_to_process)
        print(f'מעבד {len(doctor_cases_df)} מקרים עבור ד"ר {doctor_name}.')

        # --- Generate Raw Data CSV for the doctor ---
        raw_data_filename = os.path.join(OUTPUT_RAW_DATA_FOLDER, f"{doctor_name.replace(' ', '_')}_raw_data.csv")
        try:
            # Drop the temporary 'doctor_name' column before saving
            doctor_cases_df.drop(columns=['doctor_name']).to_csv(raw_data_filename, index=False, encoding='utf-8-sig')
            print(f'--- נתוני הגלם עבור ד"ר {doctor_name} נשמרו בקובץ: \'{raw_data_filename}\' ---')
        except Exception as e:
            print(f'שגיאה בכתיבת קובץ הנתונים הגולמיים עבור ד"ר {doctor_name}: {e}')
            continue # Skip to the next doctor

        # --- Prepare and Invoke LLM ---
        user_prompt = prepare_llm_user_prompt(doctor_name, doctor_cases_df)
        if not user_prompt:
            print(f'אין מקרים לנתח עבור ד"ר {doctor_name}. מדלג.')
            continue

        print(f'--- שולח בקשה ל-LLM עבור ניתוח-על של ד"ר {doctor_name} ---')
        llm_analysis_result = invoke_llm(
            system_prompt=SYSTEM_PROMPT_CROSS_ANALYSIS_HE,
            user_prompt_text=user_prompt,
            max_tokens=4000,
            temperature=0.2,
        )

        if llm_analysis_result.startswith("ERROR:"):
            print(f'ניתוח ה-LLM עבור ד"ר {doctor_name} נכשל: {llm_analysis_result}')
        else:
            print(f'--- ניתוח-על מ-LLM עבור ד"ר {doctor_name} ---')
            print(llm_analysis_result)
            print(f'--- סוף ניתוח עבור ד"ר {doctor_name} ---')

    print("\n--- הסקריפט סיים את עבודתו ---")


if __name__ == "__main__":
    main()