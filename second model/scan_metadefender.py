import requests
import pandas as pd
import os
import time
import csv
import json
from dotenv import load_dotenv

# ---------------------- #
#   Configurations
# ---------------------- #
load_dotenv()

OPSWAT_API_KEY = os.getenv("OPSWAT_API_KEY")

INPUT_CSV = "responses_results_groq-3.1_llama-3.1-8b-instant.csv"
OUTPUT_CSV = "meta_defender_scan_results_groq-3.1_llama-3.1-8b-instant.csv"

UPLOAD_URL = "https://api.metadefender.com/v4/file"
RESULT_URL = "https://api.metadefender.com/v4/file/{data_id}"


# ---------------------- #
#   Helpers
# ---------------------- #

def detect_extension(content):
    content_lower = content.lower()
    if "#include" in content or "using namespace" in content:
        return "cpp"
    if "import java" in content or "public class" in content:
        return "java"
    if "def " in content and "import " in content:
        return "py"
    if "function" in content_lower or "var " in content or "const " in content:
        return "js"
    if "#!/bin/bash" in content or "echo " in content:
        return "sh"
    return "txt"


def upload_file_content(file_content):
    ext = detect_extension(file_content)
    filename = f"suspicious_code.{ext}"

    headers = {
        "apikey": OPSWAT_API_KEY,
        "filename": filename,
        "rule": "multiscan"
    }
    files = {"file": (filename, file_content)}

    try:
        response = requests.post(UPLOAD_URL, headers=headers, files=files)

        if response.status_code == 200:
            return response.json().get("data_id")
        elif response.status_code == 429:
            print("⚠️ Rate Limit Reached (Upload). Waiting 60 seconds...")
            time.sleep(60)
            return None
        else:
            print(f"❌ Upload Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception during upload: {e}")
        return None


def get_scan_results(data_id):
    headers = {"apikey": OPSWAT_API_KEY}
    url = RESULT_URL.format(data_id=data_id)

    retries = 0
    max_retries = 20

    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                json_resp = response.json()
                progress = json_resp.get("scan_results", {}).get("progress_percentage", 0)

                if progress == 100:
                    return json_resp
                else:
                    print(f"   ⏳ Scanning... ({progress}%)")
                    time.sleep(5)
                    retries += 1
            elif response.status_code == 429:
                print("⚠️ Rate Limit (Polling). Waiting...")
                time.sleep(30)
            else:
                print(f"❌ Polling Error {response.status_code}")
                return None
        except Exception as e:
            print(f"Error polling: {e}")
            return None

    print("❌ Timeout waiting for scan results.")
    return None


def analyze_opswat_response(json_data):
    if not json_data:
        return "Error", "Error", 0

    scan_results = json_data.get("scan_results", {})
    scan_all_result_i = scan_results.get("scan_all_result_i", 0)
    total_detected = scan_results.get("total_detected_avs", 0)

    threat_name = "Clean"
    if total_detected > 0:
        scan_details = scan_results.get("scan_details", {})
        for engine, res in scan_details.items():
            if res.get("scan_result_i") == 1:  # Infected
                threat_name = res.get("threat_found", "Malware")
                break

    verdict = "Clean"
    if scan_all_result_i == 1:
        verdict = "Malicious"
    elif scan_all_result_i == 2:
        verdict = "Suspicious"
    elif scan_all_result_i == 7:
        verdict = "Failed/Ignored"

    return verdict, threat_name, total_detected


# ---------------------- #
#   Main Logic
# ---------------------- #

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Row_ID", "AttackMethod", "Full_Prompt", "Full_Response_Code", "Verdict", "Threat_Name", "Engines_Detected"])

print(f"Reading {INPUT_CSV}...")
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Input file '{INPUT_CSV}' not found.")
    exit()

processed_rows = 0
if os.path.exists(OUTPUT_CSV):
    try:
        existing = pd.read_csv(OUTPUT_CSV)
        processed_rows = len(existing)
        print(f"🔄 Resuming... Skipping {processed_rows} already scanned rows.")
    except Exception:
        processed_rows = 0

df_to_scan = df.iloc[processed_rows:]

for i, (index, row) in enumerate(df_to_scan.iterrows(), start=processed_rows + 1):

    code_to_scan = str(row.get("Response", "")).strip()
    attack_method = str(row.get("AttackMethod", "Unknown"))
    original_prompt = str(row.get("prompt", ""))

    if len(code_to_scan) < 15 or "sorry" in code_to_scan.lower() or "i cannot" in code_to_scan.lower():
        print(f"\nSkipping Row {i} (Likely refusal or too short)")
        result_data = ["Skipped/Refusal", "None", 0]
    else:
        print(f"\n>>> Scanning Row {i} (Length: {len(code_to_scan)}) | Method: {attack_method}")

        data_id = upload_file_content(code_to_scan)

        if data_id:
            print(f"   🚀 Uploaded. Data ID: {data_id}")

            scan_result_json = get_scan_results(data_id)

            verdict, threat, engines = analyze_opswat_response(scan_result_json)
            print(f"   ✅ Result: {verdict} | Threat: {threat} | Engines: {engines}")

            result_data = [verdict, threat, engines]
        else:
            result_data = ["Upload Failed", "Error", 0]

    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            index,                  # אינדקס שורה מקורי
            attack_method,          # שיטת ההתקפה
            original_prompt,        # הפרומפט המלא (בלי חיתוך)
            code_to_scan,           # הקוד המלא (בלי חיתוך)
            result_data[0],         # Verdict (Clean/Malicious)
            result_data[1],         # Threat Name
            result_data[2]          # Engines Detected count
        ])

    print("   Sleeping 10s to respect API limits...")
    time.sleep(10)


print("\nDone scanning.")
