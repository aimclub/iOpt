"""
Скрипт для запуска всех тестов и примеров
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

IOPT_ROOT = Path(__file__).parent.parent 

SCAN_DIRS = [
    IOPT_ROOT / "test",
    IOPT_ROOT / "examples",
]

LOG_FILE = Path(__file__).parent / "run_results.txt"

TIMEOUT = 2000


def find_python_files(root_dirs):
    py_files = []
    current_script = Path(__file__).resolve()
    
    IGNORED_NAMES = {
        "__init__.py",
        "__main__.py",
        "setup.py",
        "conftest.py",
        "run_all_examples.py",
    }
    
    for root in root_dirs:
        if not root.exists():
            print(f"[WARN] Директория не найдена: {root}")
            continue
        for path in root.rglob("*.py"):
            if path.resolve() == current_script:
                continue
            if path.name in IGNORED_NAMES:
                continue
            if path.name.startswith("__") and path.name.endswith(".py"):
                continue
            py_files.append(path)
    return sorted(py_files)


def run_script(script_path, cwd):
    env = os.environ.copy()

    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"

    env["PYTHONPATH"] = str(IOPT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
        
        if result.returncode == 0:
            return "OK", result.stdout + "\n" + result.stderr
        else:
            return "ERROR", (
                f"Return code: {result.returncode}\n"
                f"--- STDOUT ---\n{result.stdout}\n"
                f"--- STDERR ---\n{result.stderr}"
            )
    
    except subprocess.TimeoutExpired:
        return "ERROR", f"TIMEOUT: скрипт выполнялся дольше {TIMEOUT} секунд"
    
    except Exception as e:
        return "ERROR", f"EXCEPTION при запуске: {e}"


def main():
    print("=" * 70)
    print("ЗАПУСК")
    print("=" * 70)
    print(f"Корневая папка: {IOPT_ROOT}")
    print(f"Лог-файл: {LOG_FILE}")
    print(f"Таймаут на скрипт: {TIMEOUT} сек")
    print("=" * 70)
    
    py_files = find_python_files(SCAN_DIRS)
    print(f"\nВсего файлов: {len(py_files)}\n")
    
    if not py_files:
        print("[WARN] Нет примеров для выполнения, проверьте корректность пути.")
        return
    
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write(f"# Результаты запуска примеров iOpt\n")
        log.write(f"# Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"# Всего файлов: {len(py_files)}\n")
        log.write("=" * 70 + "\n\n")
        
        ok_count = 0
        error_count = 0
        
        for i, script_path in enumerate(py_files, 1):
            rel_path = script_path.relative_to(IOPT_ROOT)
            print(f"[{i}/{len(py_files)}] Запуск: {rel_path}")
            
            start_time = time.time()
            status, output = run_script(script_path, cwd=script_path.parent)
            elapsed = time.time() - start_time
            
            log.write(f"{'=' * 70}\n")
            log.write(f"Файл: {rel_path}\n")
            log.write(f"Статус: {status}\n")
            log.write(f"Время: {elapsed:.2f} сек\n")
            log.write(f"{'=' * 70}\n")
            
            if status == "OK":
                ok_count += 1
                print(f"[V]OK ({elapsed:.2f} сек)")
                log.write("Результат: OK\n\n")
            else:
                error_count += 1
                print(f"[X]ERROR ({elapsed:.2f} сек)")
                log.write("Результат: ERROR\n")
                log.write("--- ЛОГ ОШИБКИ ---\n")
                log.write(output + "\n\n")
            
            log.flush()
        
        log.write("\n" + "=" * 70 + "\n")
        log.write("ИТОГО:\n")
        log.write(f"  Успешно: {ok_count}\n")
        log.write(f"  Ошибок:  {error_count}\n")
        log.write(f"  Всего:   {len(py_files)}\n")
        log.write("=" * 70 + "\n")
        
        print("\n" + "=" * 70)
        print("ИТОГО:")
        print(f"  Успешно: {ok_count}")
        print(f"  Ошибок:  {error_count}")
        print(f"  Всего:   {len(py_files)}")
        print(f"\nЛог записан в: {LOG_FILE}")
        print("=" * 70)


if __name__ == "__main__":
    main()
    