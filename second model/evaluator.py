import os
import pandas as pd
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset
import time

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# for save before cancul
CHECKPOINT_FILE = "eval_checkpoint.csv"


def run_evaluator():
    os.environ["DATASETS_NUM_PROC"] = "1"
    os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

    df = pd.read_csv("responses_results_perplexity_sonar.csv")
    df = df.rename(columns={
        "prompt": "forbidden_prompt",
        "Response": "response",
        "AttackMethod": "attack_method"
    })

    if "row_id" not in df.columns:
        df.reset_index(names="row_id", inplace=True)

    df['row_id'] = df['row_id'].astype(int)

    dataset = Dataset.from_pandas(df, preserve_index=False)

    processed_ids = []
    if os.path.exists(CHECKPOINT_FILE):
        try:
            existing_data = pd.read_csv(CHECKPOINT_FILE)
            if "row_id" in existing_data.columns:
                processed_ids = existing_data["row_id"].astype(int).tolist()
            print(f"Found checkpoint file! Resuming. Already processed {len(processed_ids)} rows.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint file: {e}")

    print(f"Starting evaluation for {len(dataset)} rows...")

    judge_models = ["openai/llama-3.1-8b-instant"]

    for i in range(len(dataset)):
        current_row_id = int(dataset[i]['row_id'])

        if current_row_id in processed_ids:
            continue

        print(f"--- Processing row {i + 1} of {len(dataset)} (ID: {current_row_id}) ---")
        single_row_dataset = dataset.select([i])

        try:
            evaluated = evaluate_dataset(
                single_row_dataset,
                ["strongreject_rubric"],
                models=judge_models
            )

            # convert DataFrame
            res_df = evaluated.to_pandas()

            if "row_id" not in res_df.columns:
                res_df["row_id"] = current_row_id

            write_header = not os.path.exists(CHECKPOINT_FILE)
            res_df.to_csv(CHECKPOINT_FILE, mode='a', header=write_header, index=False)

            print(f"Row {i + 1} saved to checkpoint.")

        except Exception as e:
            print(f"Error processing row {i + 1}: {e}")
            pass

        print("Waiting 2 seconds...")
        time.sleep(2)

    print("--- Evaluation loop finished. Merging final results ---")

    if not os.path.exists(CHECKPOINT_FILE):
        print("No results found in checkpoint file. Exiting.")
        return

    ev_df = pd.read_csv(CHECKPOINT_FILE)
    ev_df['row_id'] = ev_df['row_id'].astype(int)

    cols_to_drop = [c for c in ev_df.columns if c in df.columns and c != "row_id"]
    out = df.merge(ev_df.drop(columns=cols_to_drop), on="row_id", how="left")

    out.to_csv("responses_results_evaluated.csv", index=False)
    print("Saved FINAL file -> responses_results_evaluated.csv")

# the statistic of the results
#------------------------------------------------------------------------------#
def run_statistic():
    # Load the final evaluated results
    try:
        final_df = pd.read_csv("responses_results_evaluated.csv")
    except FileNotFoundError:
        print("Error: 'responses_results_evaluated.csv' not found. Run the evaluator first!")
        return

    # 1. Calculation of Overall Jailbreak Rate (Assuming refusal=0 means Jailbreak)
    total_prompts = len(final_df)

    if 'refusal' not in final_df.columns:
        print("Error: The required 'refusal' column is missing from the evaluated results.")
        return

    # Assuming 0 is NO refusal / Jailbreak success
    jailbreaks = final_df[final_df['refusal'] == 0]
    jailbreak_rate = len(jailbreaks) / total_prompts

    print(f"\n--- Statistical Results ---")
    print(f"Total Evaluated Prompts: {total_prompts}")
    print(f"Overall Jailbreak Rate (Success of Attack): {jailbreak_rate:.2%}")
    print(f"Overall Average Safety Score: {final_df['score'].mean():.3f}")

    # 2. Calculating Effectiveness per Attack Method
    attack_stats = final_df.groupby('attack_method').agg(
        Avg_Score=('score', 'mean'),
        Total_Prompts=('row_id', 'count'),
        # Count where refusal is 0
        Jailbreak_Success_Rate=('refusal', lambda x: (x == 0).sum() / len(x))
    ).sort_values(by='Jailbreak_Success_Rate', ascending=False)

    print("\n--- Attack Method Effectiveness ---")
    print(attack_stats)
#-------------------------------------------------------------------------------#

if __name__ == "__main__":
    run_evaluator()
    run_statistic()

