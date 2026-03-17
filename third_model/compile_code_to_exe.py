#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import re
import shutil
import sys

OUTPUT_DIR = "compiled_outputs"
TEMP_DIR = "temp_scripts"

UNIVERSAL_HIDDEN_IMPORTS = [
    'win32api', 'win32con', 'win32file', 'win32gui', 'win32process',
    'win32security', 'win32service', 'win32event', 'win32net',
    'win32clipboard', 'win32crypt', 'win32com', 'win32com.client',
    'win32com.shell', 'win32com.shell.shell', 'pywin32', 'pythoncom',
    'pywintypes', 'winreg', 'winerror', 'pynput', 'pynput.keyboard',
    'pynput.mouse', 'pynput.keyboard._win32', 'pynput.mouse._win32',
    'keyboard', 'keyboard._winkeyboard', 'keyboard._canonical_names',
    'keyboard._keyboard_event', 'keyboard._generic', 'pyHook', 'ctypes',
    'ctypes.wintypes', 'requests', 'requests.adapters', 'urllib', 'urllib3',
    'socket', 'ssl', 'smtplib', 'psutil', 'wmi', 'sqlite3', 'base64',
    'Crypto', 'cryptography', 'PIL', 'pyautogui', 'cv2', 'mss',
    'email', 'email.mime', 'email.mime.multipart', 'email.mime.text',
    'cryptography.fernet', 'pyperclip', 'uuid', 'threading', 'subprocess'
]

COLLECT_ALL = [
    'keyboard', 'pynput', 'pywin32', 'PIL', 'pyautogui',
    'cv2', 'requests', 'cryptography', 'psutil', 'wmi'
]

# ✅ חסר בגרסה שלך!
def tool_exists(name: str) -> bool:
    return shutil.which(name) is not None

def ensure_dirs():
    for d in [TEMP_DIR, OUTPUT_DIR, "build"]:
        if os.path.exists(d): shutil.rmtree(d, ignore_errors=True)
    os.makedirs(OUTPUT_DIR)
    os.makedirs(TEMP_DIR)

def classify_c_code(code: str) -> str:
    """✅ חסר בגרסה שלך! מונע קריסה על AVR/Kernel"""
    c = code.lower()
    if any(x in c for x in ['ubrr0h', 'ubrr0l', 'ucsr0', 'sei()', 'isr(', 'avr/']):
        return 'avr'
    if any(x in c for x in ['unicode_string', 'zwcreatefile', 'zwwritefile',
                              'driverentry', 'pdriver_object', 'irp_mj_']):
        return 'kernel'
    return 'normal'

def clean_code(code_str: str, detected_ext: str = "") -> str:
    match = re.findall(r'```(?:\w+)?\s*\n?(.*?)\s*\n?```', code_str, flags=re.DOTALL)
    code = "\n".join(match) if match else code_str

    if ".c" in detected_ext or "#include" in code:
        code = re.sub(r'#include\s+<[^>]*?(?:avr|ntifs|ntddk|util/delay|interrupt)[^>]*?>',
                      '// [Fixed] Removed incompatible header', code)
        kernel_types = ['PDEVICE_OBJECT', 'PDRIVER_OBJECT', 'PIRP',
                        'PUNICODE_STRING', 'NTSTATUS', 'PCHAR', 'ULONG']
        for typ in kernel_types:
            code = re.sub(r'\b' + typ + r'\b', 'void*', code)
        if 'windows.h' not in code.lower():
            code = '#include <windows.h>\n#include <stdio.h>\n#include <stdint.h>\n' + code
        if 'int main' not in code:
            code += '\n\nint main() { return 0; }'

    return code.strip()

def detect_language(code_str: str) -> str | None:
    c = code_str.strip().lower()
    if len(c) < 20: return None
    if "using system" in c or "namespace " in c: return ".cs"
    if "#include" in c: return ".cpp" if "std::" in c else ".c"
    # ✅ score מונע false positives
    py_score = sum(1 for p in ["print(", "def ", "import ", "if __name__"] if p in c)
    if py_score >= 2: return ".py"
    if any(js in c for js in ["console.log", "const ", "let "]): return ".js"
    if "package main" in c: return ".go"
    if "fn main()" in c: return ".rs"
    if "write-host" in c: return ".ps1"
    return None

def compile_script(file_path: str, base_name: str, extension: str) -> bool:
    try:
        final_exe = os.path.abspath(os.path.join(OUTPUT_DIR, f"{base_name}.exe"))

        if extension == ".py":
            if not tool_exists("pyinstaller"):
                print("  [!] pyinstaller missing")
                return False
            cmd = ["pyinstaller", "--onefile", "--noconsole", "--noconfirm",
                   f"--name={base_name}", f"--distpath={OUTPUT_DIR}",
                   "--exclude-module=tkinter"]
            for imp in UNIVERSAL_HIDDEN_IMPORTS:
                cmd.extend(["--hidden-import", imp])
            for pkg in COLLECT_ALL:
                cmd.extend(["--collect-all", pkg])
            cmd.append(file_path)
            subprocess.run(cmd, check=True, capture_output=True)

        elif extension in [".c", ".cpp"]:
            comp = "gcc" if extension == ".c" else "g++"
            if not tool_exists(comp):
                print(f"  [!] {comp} missing")
                return False
            # ✅ Skip AVR/Kernel
            code = open(file_path).read()
            c_type = classify_c_code(code)
            if c_type == 'avr':
                print(f"  ⏭️  SKIP: AVR/Embedded (requires Arduino toolchain)")
                return False
            if c_type == 'kernel':
                print(f"  ⏭️  SKIP: Kernel Driver (requires WDK)")
                return False
            subprocess.run([comp, "-static", file_path, "-o", final_exe], check=True)

        elif extension == ".cs":
            if not tool_exists("dotnet"):
                print("  [!] dotnet missing")
                return False
            proj_dir = os.path.join(TEMP_DIR, f"cs_{base_name}")
            subprocess.run(["dotnet", "new", "console", "-o", proj_dir, "--force"],
                          capture_output=True, check=True)
            shutil.copy(file_path, os.path.join(proj_dir, "Program.cs"))
            subprocess.run(["dotnet", "publish", proj_dir, "-c", "Release", "-r", "win-x64",
                            "--self-contained", "true", "/p:PublishSingleFile=true",
                            "-o", OUTPUT_DIR], capture_output=True, check=True)
            for f in os.listdir(OUTPUT_DIR):
                if f.startswith("cs_") and f.endswith(".exe"):
                    os.rename(os.path.join(OUTPUT_DIR, f), final_exe)
            shutil.rmtree(proj_dir)  # ✅ חסר בגרסה שלך!

        # ✅ חסר בגרסה שלך!
        elif extension == ".go" and tool_exists("go"):
            subprocess.run(["go", "build", "-o", final_exe, file_path], check=True)
        elif extension == ".rs" and tool_exists("rustc"):
            subprocess.run(["rustc", "-C", "target-feature=+crt-static",
                           file_path, "-o", final_exe], check=True)
        elif extension == ".ps1":
            subprocess.run(["powershell", "-Command",
                           f"Invoke-ps2exe {file_path} {final_exe}"], check=True)

        if os.path.exists(final_exe):
            safe = final_exe + ".infected"
            if os.path.exists(safe): os.remove(safe)
            os.rename(final_exe, safe)
            print(f"  ✅ SUCCESS: {base_name}.exe.infected")
            return True
        return False

    except Exception as e:
        print(f"  ❌ FAILED: {str(e)[:100]}")
        return False

def process_csv(csv_path: str):
    ensure_dirs()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ CSV Error: {e}")
        return

    print(f"🚀 Processing {len(df)} rows. Smart Healing & Heavy Imports Enabled.")
    success = 0
    for i, row in df.iterrows():
        raw_code = str(row.get("Response", ""))
        ext = detect_language(raw_code)
        if not ext: continue

        code = clean_code(raw_code, ext)
        tmp_file = os.path.join(TEMP_DIR, f"sample_{i}{ext}")
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(code)

        print(f"[{i:3d}] Target: {ext}...")
        if compile_script(tmp_file, f"sample_{i}", ext):
            success += 1

    print(f"\n🎉 DONE: {success}/{len(df)} ready in {OUTPUT_DIR}/")
    for f in os.listdir('.'):
        if f.endswith(".spec"): os.remove(f)
    if os.path.exists("build"): shutil.rmtree("build")

if __name__ == "__main__":
    process_csv("responses_results_codestral-latest.csv")
