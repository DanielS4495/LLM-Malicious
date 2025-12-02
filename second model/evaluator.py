import os
import pandas as pd
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset
import time
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS_PER_REQUEST = 8000

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

INPUT_FILE = "responses_results_perplexity_sonar.csv"
CHECKPOINT_FILE = "eval_checkpoint.csv"

ACTIVE_JUDGE = "gemini"  # Options: perplexity, groq, gemini, huggingface, openrouter


def configure_judge_environment(provider):
    os.environ.pop("OPENAI_API_BASE", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("GROQ_API_KEY", None)
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("HUGGINGFACE_API_KEY", None)
    os.environ.pop("PERPLEXITY_API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)
    model_name = ""

    if provider == "perplexity":
        if not PPLX_API_KEY: raise ValueError("Missing PPLX_API_KEY")
        os.environ["OPENAI_API_KEY"] = PPLX_API_KEY
        os.environ["OPENAI_API_BASE"] = "https://api.perplexity.ai"
        model_name = "openai/llama-3.1-sonar-large-128k-online"

    elif provider == "groq":
        if not GROQ_API_KEY: raise ValueError("Missing GROQ_API_KEY")
        os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
        os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
        model_name = "openai/llama-3.3-70b-versatile"

    elif provider == "gemini":
        if not GEMINI_API_KEY: raise ValueError("Missing GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        model_name = "gemini/gemini-2.0-flash"

    elif provider == "huggingface":
        if not HF_TOKEN: raise ValueError("Missing HF_TOKEN")
        os.environ["HUGGINGFACE_API_KEY"] = HF_TOKEN
        model_name = "huggingface/meta-llama/Meta-Llama-3-70B-Instruct"

    elif provider == "openrouter":
        if not OPENROUTER_API_KEY: raise ValueError("Missing OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
        model_name = "openrouter/meta-llama/llama-3.3-70b-instruct:free"

    else:
        raise ValueError(f"Unknown provider: {provider}")

    print(f"✅ Configured environment for: {provider}")
    print(f"✅ Target Model: {model_name}")
    return [model_name]


def retry_failed_rows():
    """
    Identifies rows with missing data (NaN) in checkpoint and reprocesses them
    """
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"❌ Checkpoint file '{CHECKPOINT_FILE}' not found.")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file '{INPUT_FILE}' not found.")
        return

    # Read files
    checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
    input_df = pd.read_csv(INPUT_FILE)

    # Rename columns in input file
    input_df = input_df.rename(columns={
        "prompt": "forbidden_prompt",
        "Response": "response",
        "AttackMethod": "attack_method"
    })

    if "row_id" not in input_df.columns:
        input_df.reset_index(names="row_id", inplace=True)

    input_df['row_id'] = input_df['row_id'].astype(int)
    checkpoint_df['row_id'] = checkpoint_df['row_id'].astype(int)

    # Identify rows with missing data - rows that have empty score or refusal
    failed_rows = checkpoint_df[
        checkpoint_df['score'].isna() | checkpoint_df['refusal'].isna()
        ]

    if len(failed_rows) == 0:
        print("✅ No failed rows found. All rows have complete data!")
        return

    print(f"🔍 Found {len(failed_rows)} rows with missing data")
    print(f"Failed row IDs: {failed_rows['row_id'].tolist()}")

    try:
        judge_models = configure_judge_environment(ACTIVE_JUDGE)
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        return

    os.environ["DATASETS_NUM_PROC"] = "1"

    # Process each failed row
    for idx, failed_row in failed_rows.iterrows():
        row_id = int(failed_row['row_id'])

        print(f"\n--- Retrying row_id {row_id} ---")

        # Find original row from input file
        original_row = input_df[input_df['row_id'] == row_id]

        if len(original_row) == 0:
            print(f"⚠️ Warning: row_id {row_id} not found in input file. Skipping.")
            continue

        # Create dataset from single row
        single_dataset = Dataset.from_pandas(original_row, preserve_index=False)

        try:
            # Evaluate the row
            evaluated = evaluate_dataset(
                single_dataset,
                ["strongreject_rubric"],
                models=judge_models,
                max_tokens=MAX_TOKENS_PER_REQUEST
            )

            # Convert to DataFrame
            res_df = evaluated.to_pandas()

            if "row_id" not in res_df.columns:
                res_df["row_id"] = row_id

            # Update checkpoint file - remove old row and append new one
            checkpoint_df = checkpoint_df[checkpoint_df['row_id'] != row_id]
            checkpoint_df = pd.concat([checkpoint_df, res_df], ignore_index=True)
            checkpoint_df.to_csv(CHECKPOINT_FILE, index=False)

            print(f"✅ Successfully retried and updated row_id {row_id}")

        except Exception as e:
            print(f"❌ Error processing row_id {row_id}: {e}")
            if "401" in str(e) or "Authentication" in str(e):
                print("CRITICAL AUTH ERROR. Stopping retry process.")
                break

        print("Waiting 10 seconds...")
        time.sleep(10)

    print("\n--- Retry process completed ---")


def run_evaluator():
    os.environ["DATASETS_NUM_PROC"] = "1"
    try:
        judge_models = configure_judge_environment(ACTIVE_JUDGE)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    df = pd.read_csv(INPUT_FILE)
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

    print(f"Starting evaluation for {len(dataset)} rows using {ACTIVE_JUDGE}...")

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
                models=judge_models,
                max_tokens=MAX_TOKENS_PER_REQUEST
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
            if "401" in str(e) or "Authentication" in str(e):
                print("CRITICAL AUTH ERROR. Stopping.")
                break
            pass

        print("Waiting 10 seconds...")
        time.sleep(10)

    print("--- Evaluation loop finished. Merging final results ---")

    if not os.path.exists(CHECKPOINT_FILE):
        print("No results found in checkpoint file. Exiting.")
        return

    ev_df = pd.read_csv(CHECKPOINT_FILE)
    ev_df['row_id'] = ev_df['row_id'].astype(int)

    cols_to_drop = [c for c in ev_df.columns if c in df.columns and c != "row_id"]
    out = df.merge(ev_df.drop(columns=cols_to_drop), on="row_id", how="left")

    out.to_csv("responses_results_evaluated_preplexity_sonar.csv", index=False)
    print("Saved FINAL file -> responses_results_evaluated_preplexity_sonar.csv")


def run_statistic():
    try:
        final_df = pd.read_csv("responses_results_evaluated_preplexity_sonar.csv")
    except FileNotFoundError:
        print("Error: 'responses_results_evaluated_preplexity_sonar.csv' not found. Run the evaluator first!")
        return

    total_prompts = len(final_df)

    # delete those without score
    final_df = final_df.dropna(subset=['score'])

    if len(final_df) == 0:
        print("No valid scores found.")
        return

    if 'refusal' not in final_df.columns:
        print("Error: 'refusal' column missing.")
        return

    # Assuming 0 is NO refusal / Jailbreak success
    jailbreaks = final_df[final_df['refusal'] == 0]
    jailbreak_rate = len(jailbreaks) / total_prompts

    print(f"\n--- Statistical Results ---")
    print(f"Total Evaluated Prompts: {total_prompts}")
    print(f"Overall Jailbreak Rate (Success of Attack): {jailbreak_rate:.2%}")
    print(f"Overall Average Safety Score: {final_df['score'].mean():.3f}")

    if 'attack_method' in final_df.columns:
        attack_stats = final_df.groupby('attack_method').agg(
            Avg_Score=('score', 'mean'),
            Total_Prompts=('row_id', 'count'),
            Jailbreak_Success_Rate=('refusal', lambda x: (x == 0).sum() / len(x))
        ).sort_values(by='Jailbreak_Success_Rate', ascending=False)

        print("\n--- Attack Method Effectiveness ---")
        print(attack_stats)


if __name__ == "__main__":
    import sys

    # Check if user wants to retry failed rows
    if len(sys.argv) > 1 and sys.argv[1] == "--retry":
        print("🔄 RETRY MODE: Processing failed rows only")
        retry_failed_rows()
        # Regenerate final results after retry
        print("\n--- Merging results after retry ---")
        if os.path.exists(CHECKPOINT_FILE):
            df = pd.read_csv(INPUT_FILE)
            df = df.rename(columns={
                "prompt": "forbidden_prompt",
                "Response": "response",
                "AttackMethod": "attack_method"
            })
            if "row_id" not in df.columns:
                df.reset_index(names="row_id", inplace=True)
            df['row_id'] = df['row_id'].astype(int)

            ev_df = pd.read_csv(CHECKPOINT_FILE)
            ev_df['row_id'] = ev_df['row_id'].astype(int)
            cols_to_drop = [c for c in ev_df.columns if c in df.columns and c != "row_id"]
            out = df.merge(ev_df.drop(columns=cols_to_drop), on="row_id", how="left")
            out.to_csv("responses_results_evaluated_preplexity_sonar.csv", index=False)
            print("Updated final file -> responses_results_evaluated_preplexity_sonar.csv")

        run_statistic()
    else:
        print("▶️ NORMAL MODE: Running full evaluation")
        run_evaluator()
        run_statistic()