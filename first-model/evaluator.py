import os
import pandas as pd
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset
import time
GROQ_API_KEY = os.getenv("GROQ_API_KEY") #need to add a key
def run_evaluator():


    # can be change to openai key or huggging face
    #os.environ["OPENAI_API_KEY"] = OPENAI_KEY
    #os.environ["HUGGINGFACE_API_KEY"] = HF_TOKEN
    os.environ["DATASETS_NUM_PROC"] = "1"
    os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

    df = pd.read_csv("responses_results.csv")
    df = df.rename(columns={
        "prompt": "forbidden_prompt",
        "Response": "response",
        "AttackMethod": "attack_method"
    })
    if "row_id" not in df.columns:
        df.reset_index(names="row_id", inplace=True)

    if "jailbroken_prompt" not in df.columns and "forbidden_prompt" in df.columns:
        df["jailbroken_prompt"] = df["forbidden_prompt"]

    dataset = Dataset.from_pandas(df, preserve_index=False)
    all_eval_dfs = []
    print(f"Starting evaluation for {len(dataset)} rows (one by one)...")
    # שליטה במודל השופט (נסה לפי סדר)
    #judge_models = ["openai/gpt-4o-mini", "openai/gpt-4o"]  # ניתן להחליף/להרחיב [web:56][web:43]
    #judge_models = ["huggingface/meta-llama/Llama-3-8B-Instruct"]  # ⬅️ התיקון
    judge_models = ["openai/llama-3.1-8b-instant"]  # ⬅️ התיקון (השם המעודכן)
    # ⬅️ 4. הלולאה החדשה - עוברת על כל שורה בדאטהסט
    for i in range(len(dataset)):
        print(f"--- Processing row {i + 1} of {len(dataset)} ---")

        # ניקח רק את השורה הנוכחית
        single_row_dataset = dataset.select([i])

        try:
            # נריץ את הבדיקה רק על השורה הזו
            evaluated = evaluate_dataset(
                single_row_dataset,
                ["strongreject_rubric"],
                models=judge_models
            )

            # נוסיף את התוצאה (שהיא DataFrame) לרשימה
            all_eval_dfs.append(evaluated.to_pandas())
            print(f"Row {i + 1} evaluated successfully.")

        except Exception as e:
            print(f"Error processing row {i + 1}: {e}")
            # אפשר להוסיף כאן לוגיקה לשמירת שורות שנכשלו
            # כרגע פשוט נדלג
            pass

            # ⬅️ 5. ההשהייה! נחכה 2 שניות לפני הבקשה הבאה
        # אפשר להגדיל ל-3 או 5 אם הבעיה חוזרת
        print("Waiting 2 seconds to avoid rate limit...")
        time.sleep(2)

    print("--- Evaluation finished, merging results ---")
    # ⬅️ 6. נאחד את כל תוצאות ה-DataFrame שקיבלנו
    if not all_eval_dfs:
        print("No evaluations were successful. Exiting.")
        return

    ev_df = pd.concat(all_eval_dfs, ignore_index=True)

    # ⬅️ 7. נמזג עם ה-DataFrame המקורי ונשמור

    # ========== התיקון: כופים התאמת סוג נתונים ==========
    print("Casting row_id columns to int before merge...")
    df['row_id'] = df['row_id'].astype(int)
    ev_df['row_id'] = ev_df['row_id'].astype(int)
    # ================================================

    # ודא שעמודת row_id קיימת ומדויקת
    cols_to_drop = [c for c in ev_df.columns if c in df.columns and c != "row_id"]

    out = df.merge(ev_df.drop(columns=cols_to_drop), on="row_id", how="left")

    out.to_csv("responses_results_evaluated.csv", index=False)
    print("Saved -> responses_results_evaluated.csv")


if __name__ == "__main__":
    run_evaluator()