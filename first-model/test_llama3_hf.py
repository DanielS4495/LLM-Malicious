import pandas as pd
import os
from huggingface_hub import InferenceClient
import os, getpass

# Load Excel
df = pd.read_excel('attack_prompts.xlsx')


HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = getpass.getpass("Enter your Hugging Face token: ")
client = InferenceClient(token=HF_TOKEN)


# Filter by specific criteria
# Example: Only specific attack method
df_filtered = df[df['AttackMethod'] == 'Persuative LLM']

print(f"Filtered to {len(df_filtered)} prompts")

temp = df_filtered.head()

# Initialize results list for new file
results = []

# Initialize Response column in original dataframe if it doesn't exist
if 'Response' not in df.columns:
    df['Response'] = ''

# Process each prompt
for index, row in temp.iterrows():
    prompt_text = row['prompt']
    behaviour = row['AttackMethod']
    print(f"\nBehavior: {behaviour}")
    
    response_text = None
    try:
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=4096 
        )
        
        response_text = response.choices[0].message.content
        print(f"\nResponse:")
        print(response_text)
        
        # Update the original dataframe
        df.at[index, 'Response'] = response_text
        
        # Add to results list for new file
        results.append({
            'AttackMethod': behaviour,
            'prompt': prompt_text,
            'Response': response_text
        })
            
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        
        # Update the original dataframe with error
        df.at[index, 'Response'] = error_msg
        
        # Add to results list with error
        results.append({
            'AttackMethod': behaviour,
            'prompt': prompt_text,
            'Response': error_msg
        })
        
    print("-" * 70)

# Save results to new file (append mode)
results_file = 'responses_results.csv'
if os.path.exists(results_file):
    # Append to existing file
    existing_df = pd.read_csv(results_file)
    new_df = pd.DataFrame(results)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(results_file, index=False)
    print(f"\nAppended {len(results)} results to {results_file}")
else:
    # Create new file
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"\nCreated new file {results_file} with {len(results)} results")

# Update the original Excel file
df.to_excel('attack_prompts.xlsx', index=False)
print(f"Updated original Excel file with responses")
