import pandas as pd
import os
from huggingface_hub import InferenceClient
import time
import csv
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from openai import OpenAI

# --- Settings ---
#HF_TOKEN = os.getenv("HF_TOKEN")
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#PPLX_API_KEY = os.getenv("PPLX_API_KEY")
#if not HF_TOKEN:
#    print("HF_TOKEN not found! Please set your environment variable.")

EXCEL_FILE = 'attack_prompts.xlsx'
RESULTS_FILE = 'responses_results.csv'
FIELDNAMES = ['AttackMethod', 'prompt', 'Response']

# --- Create CSV file if it doesn't exist ---
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
    print(f"Created new results file: {RESULTS_FILE}")

#client = InferenceClient(token=HF_TOKEN)
#client = Groq(api_key=GROQ_API_KEY)
client = OpenAI(api_key=PPLX_API_KEY,base_url="https://api.perplexity.ai/")

# --- Load Excel (Read Only) ---
print(f"Reading Excel file: {EXCEL_FILE}...")
df = pd.read_excel(EXCEL_FILE)

# Filtering (if needed)
df_filtered = df[df['AttackMethod'] == 'Persuative LLM']

total_rows = len(df_filtered)
print(f"Starting process for {total_rows} prompts...")

# --- Main Loop ---
for i, (index, row) in enumerate(df_filtered.iterrows()):

    # Start timer
    start_time = time.time()

    prompt_text = row['prompt']
    behaviour = row['AttackMethod']

    print(f"\n>>> Processing Row {i + 1}/{total_rows} (Original Index: {index})")

    # Clean the prompt (Manual simple method)
    original_prompt = prompt_text  # Keep original just in case
    prompt_text = str(prompt_text).strip()
    prompt_text = prompt_text.replace('[', '').replace(']', '')  # Remove brackets
    prompt_text = prompt_text.strip('"').strip("'")  # Remove surrounding quotes
    prompt_text = prompt_text.replace('\\"', '"').replace('\\n', '\n')  # Fix escaped characters

    try:
        # Send to model

        #Hugging face
        #------------------------------------------------------#
        #response = client.chat_completion(
        #    model="meta-llama/Meta-Llama-3-70B-Instruct",
        #    messages=[{"role": "user", "content": prompt_text}],
        #    max_tokens=4096
        #)
        #------------------------------------------------------#

        #GROQ + PREPLEXITY
        #----------------------------------------------------------------#
        response = client.chat.completions.create(
            model="sonar",
            #model="llama3-70b-8192",   #groq
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=4096
        )
        # ----------------------------------------------------------------#

        response_text = response.choices[0].message.content

        # Save to CSV (only if successful)
        result_data = {
            'AttackMethod': behaviour,
            'prompt': prompt_text,  # Save the clean prompt
            'Response': response_text
        }

        with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(result_data)

        # Calculate duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"✅ Success! Saved to CSV. (Time taken: {duration:.2f} seconds)")
    except Exception as e:
        # Error handling (Print to terminal only)
        end_time = time.time()
        duration = end_time - start_time
        print(f"   ERROR on Row {index}: {e}")
        print(f"   [DEBUG REPORT]")
        print(f"   1. Length of prompt: {len(prompt_text)}")
        print(f"   2. Type of data: {type(prompt_text)}")
        print(f"   3. Content (RAW): {repr(prompt_text)}")
        print(f"   (Time taken: {duration:.2f} seconds)")
        print("-" * 30)
        # Do not save anything to CSV
    time.sleep(2)

print(f"\nDone. All successful responses saved to {RESULTS_FILE}.")
