import os
import subprocess
import pandas as pd
import re
import shutil

OUTPUT_DIR = "compiled_outputs"
TEMP_DIR = "temp_scripts"

def ensure_dirs():
    """Removes old temp files and ensures output directories exist."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    for d in [OUTPUT_DIR, TEMP_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

def clean_code(code_str: str) -> str:
    """Removes Markdown code blocks and strips whitespace."""
    return re.sub(r'```(?:\w+)?\n(.*?)\n```', r'\1', code_str, flags=re.DOTALL).strip()

def _looks_like_python(c: str) -> bool:
    if "if __name__ == \"__main__\"" in c: return True
    if re.search(r"def\s+\w+\s*\(.*\)\s*:", c): return True
    if re.search(r"class\s+\w+\s*\(?.*?\)?:", c): return True
    if "import " in c and ("print(" in c or "range(" in c): return True
    return False

def detect_language(code_str: str) -> str or None:
    c = code_str.strip().lower()
    if not c: return None

    # System & Shell
    if c.startswith("#!/bin/bash") or c.startswith("#!/bin/sh"): return ".sh"
    if "write-host" in c or "set-executionpolicy" in c or "$psversiontable" in c: return ".ps1"
    if c.startswith("@echo off") or " set /p " in c: return ".bat"

    # Compiled Languages
    if "fn main()" in c and ("println!" in c or "use std::" in c): return ".rs"
    if "package main" in c and "func main(" in c: return ".go"
    if "using system;" in c or "static void main" in c: return ".cs"
    if "#include" in c:
        return ".cpp" if ("std::" in c or "iostream" in c) else ".c"
    if "public class" in c and "static void main" in c: return ".java"
    
    # Scripting & Web
    if "<?php" in c: return ".php"
    if "console.log" in c or "const " in c or "let " in c:
        return ".ts" if (": string" in c or "interface " in c) else ".js"
    if "def " in c and "end" in c and ("puts " in c or "class " in c): return ".rb"
    if "function " in c and "end" in c and ("local " in c or "then" in c): return ".lua"
    
    # Data & Markup (Skipped for EXE)
    if "<html" in c: return ".html"
    if "select " in c and "from " in c: return ".sql"
    if c.startswith("{") and ":" in c: return ".json"

    # Fallback to Python
    if _looks_like_python(c): return ".py"
    return None

def compile_script(file_path: str, base_name: str, extension: str) -> bool:
    try:
        out_exe = os.path.abspath(os.path.join(OUTPUT_DIR, f"{base_name}.exe"))
        
        # PYTHON
        if extension == ".py":
            subprocess.run(["pyinstaller", "--onefile", "--noconsole", "--distpath", OUTPUT_DIR, file_path], 
                           check=True, capture_output=True)

        # C / C++
        elif extension in [".c", ".cpp"]:
            comp = "gcc" if extension == ".c" else "g++"
            subprocess.run([comp, "-static", file_path, "-o", out_exe], check=True)

        # JS / TS (Requires 'pkg' installed via npm)
        elif extension in [".js", ".ts"]:
            subprocess.run(["pkg", file_path, "--targets", "node16-win-x64", "--output", out_exe], check=True)

        # POWERSHELL (Requires 'ps2exe' module)
        elif extension == ".ps1":
            subprocess.run(["powershell", "-Command", f"Invoke-ps2exe {file_path} {out_exe}"], check=True)

        # C# (.NET Core/6+)
        elif extension == ".cs":
            proj = os.path.join(TEMP_DIR, f"proj_{base_name}")
            subprocess.run(["dotnet", "new", "console", "-o", proj], check=True, capture_output=True)
            shutil.copy(file_path, os.path.join(proj, "Program.cs"))
            subprocess.run(["dotnet", "publish", proj, "-c", "Release", "-r", "win-x64", 
                            "--self-contained", "true", "/p:PublishSingleFile=true", "-o", OUTPUT_DIR], check=True)
            # Find and rename the published EXE
            actual_out = os.path.join(OUTPUT_DIR, f"proj_{base_name}.exe")
            if os.path.exists(actual_out):
                os.rename(actual_out, out_exe)

        # GO
        elif extension == ".go":
            subprocess.run(["go", "build", "-o", out_exe, file_path], check=True)

        # RUST
        elif extension == ".rs":
            subprocess.run(["rustc", "-C", "target-feature=+crt-static", file_path, "-o", out_exe], check=True)

        # BATCH (Wrap into a simple runner if needed, otherwise skip)
        elif extension == ".bat":
            print(f"[*] Batch file {base_name} saved. Consider using 'iexpress' for true EXE wrapping.")
            return True

        else:
            print(f"[*] Skipping {extension} - no EXE mapping defined.")
            return True

        return True
    except Exception as e:
        print(f"[-] Error compiling {base_name} ({extension}): {e}")
        return False

def process_csv(csv_path: str):
    ensure_dirs()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[-] Failed to read CSV: {e}")
        return

    for index, row in df.iterrows():
        code_raw = str(row.get("Response", ""))
        code = clean_code(code_raw)
        ext = detect_language(code)

        if not ext:
            print(f"[!] Row {index}: Language detection failed.")
            continue

        f_path = os.path.join(TEMP_DIR, f"sample_{index}{ext}")
        with open(f_path, "w", encoding="utf-8") as f:
            f.write(code)

        print(f"[*] Processing Row {index} | Target: {ext}")
        compile_script(f_path, f"sample_{index}", ext)

    # Cleanup leftover artifacts
    if os.path.exists("build"): shutil.rmtree("build")
    for item in os.listdir('.'):
        if item.endswith(".spec"): os.remove(item)

if __name__ == "__main__":
    csv_file = input("Enter the path to your CSV file: ").strip('"')
    process_csv(csv_file)
