import pandas as pd
import os
from huggingface_hub import InferenceClient
import time  # ⬅️ 1. נייבא ספריות שצריך
import csv

# --- הגדרות ---
HF_TOKEN = os.getenv("HF_TOKEN")  # ⬅️ 2. עדיף לקרוא מمتغير סביבה
if not HF_TOKEN:
    print("אנא הגדר משתנה סביבה HF_TOKEN")
    # HF_TOKEN = "hf_..." # או הדבק כאן (פחות מאובטח)

# ⬅️ 3. הגדרת קבצים ושמות עמודות
EXCEL_FILE = 'attack_prompts.xlsx'
RESULTS_FILE = 'responses_results.csv'
FIELDNAMES = ['AttackMethod', 'prompt', 'Response']  # עמודות לקובץ ה-CSV

# --- בדיקה אם קובץ ה-CSV קיים וצריך כותרת ---
file_exists = os.path.exists(RESULTS_FILE)
if not file_exists:
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
    print(f"Created new file with headers: {RESULTS_FILE}")

client = InferenceClient(token=HF_TOKEN)

# Load Excel
df = pd.read_excel(EXCEL_FILE)

# Filter by specific criteria
df_filtered = df[df['AttackMethod'] == 'Persuative LLM']
# ⬅️ או להריץ על הכל:
# df_filtered = df
# ... (כל הקוד שלפני הלולאה נשאר זהה) ...

print(f"Filtered to {len(df_filtered)} prompts. Starting process...")

# Initialize Response column if it doesn't exist
if 'Response' not in df.columns:
    df['Response'] = ''

# ⬅️ 4. שימוש ב-enumerate כדי שנוכל לספור איטרציות (לצורך שמירה)
for i, (index, row) in enumerate(df_filtered.iterrows()):
    prompt_text = row['prompt']
    behaviour = row['AttackMethod']
    print(f"\n--- Processing row {i + 1} of {len(df_filtered)} (Index: {index}) ---")
    print(f"Behavior: {behaviour}")

    response_text = None

    try:
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=4096
        )

        response_text = response.choices[0].message.content
        print(f"\nResponse:\n{response_text}")

        # --- הגיון השמירה החדש (רק בהצלחה) ---

        # 1. הכנת נתוני ההצלחה
        result_data = {
            'AttackMethod': behaviour,
            'prompt': prompt_text,
            'Response': response_text
        }

        # 2. עדכון קובץ האקסל (בזיכרון)
        df.at[index, 'Response'] = response_text

        # 3. כתיבה *מיידית* לקובץ ה-CSV
        with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(result_data)

        print(f"Saved result for index {index} to {RESULTS_FILE}")
        # ----------------------------------------

    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        print(f"!!! FAILED ON PROMPT (Index: {index}): {prompt_text}")

        # ⬅️ עדכון קובץ האקסל עם השגיאה (כדי שנדע שנכשל)
        df.at[index, 'Response'] = error_msg

        # ⬅️ אנחנו *לא* כותבים כלום ל-CSV במקרה של שגיאה
        print(f"Skipped writing to CSV for index {index} due to error.")

    # ⬅️ 7. שמירת "נקודת ביקורת" (Checkpoint) באקסל כל 10 שורות
    if (i + 1) % 10 == 0:
        df.to_excel(EXCEL_FILE, index=False)
        print(f"\n*** CHECKPOINT: Saved Excel file at row {i + 1} ***\n")

    # ⬅️ 8. המתנה למניעת Rate Limit
    print("-" * 70)
    time.sleep(2)  # המתנה של 2 שניות

# --- סיום ---
# ⬅️ 9. שמירה סופית של האקסל אחרי שהלולאה הסתיימה
df.to_excel(EXCEL_FILE, index=False)
print(f"Finished processing. Updated original Excel file: {EXCEL_FILE}")

