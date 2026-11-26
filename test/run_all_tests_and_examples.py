"""
Скрипт для запуска всех тестов и примеров с итоговой сводкой по категориям ошибок
"""

import os
import sys
import subprocess
import time
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

IOPT_ROOT = Path(__file__).parent.parent

SCAN_DIRS = [
    IOPT_ROOT / "test",
    IOPT_ROOT / "examples",
]

LOG_FILE = Path(__file__).parent / "run_results.txt"

TIMEOUT = 2000


# ============================================================================
# Классификация ошибок
# ============================================================================

def classify_error(output: str, script_path: Path) -> str:
    """
    Определяет категорию ошибки по тексту лога.
    Возвращает строку-категорию.
    """
    # 1. Отсутствующие пакеты
    match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", output)
    if match:
        return f"Missing package: {match.group(1)}"

    # 2. Отсутствующие файлы данных
    match = re.search(r"FileNotFoundError:.*?No such file or directory: '([^']+)'", output)
    if match:
        return f"Missing file: {Path(match.group(1)).name}"

    # 3. NumPy 2.0 несовместимость
    if "np.infty" in output or "was removed in the NumPy 2.0" in output:
        return "NumPy 2.0 incompatibility (np.infty -> np.inf)"

    # 4. Атрибут не инициализирован
    match = re.search(r"AttributeError: '(\w+)' object has no attribute '(\w+)'", output)
    if match:
        return f"AttributeError: {match.group(1)}.{match.group(2)}"

    # 5. Таймаут
    if "TIMEOUT:" in output:
        return "Timeout"

    # 6. UnboundLocalError (следствие других багов)
    if "UnboundLocalError" in output:
        return "UnboundLocalError (internal bug)"

    # 7. Синтаксические / прочие
    if "SyntaxError" in output:
        return "SyntaxError"

    if "ImportError" in output:
        return "ImportError (other)"

    if "ValueError" in output:
        match = re.search(r"ValueError: ([^\n]{0,80})", output)
        msg = match.group(1) if match else ""
        return f"ValueError: {msg[:60]}"

    if "TypeError" in output:
        match = re.search(r"TypeError: ([^\n]{0,80})", output)
        msg = match.group(1) if match else ""
        return f"TypeError: {msg[:60]}"

    if "NO TESTS RAN" in output:
        return "No tests ran (unittest)"

    return "Other"


def extract_missing_packages(errors: list) -> set:
    """Собирает все отсутствующие пакеты из списка ошибок."""
    packages = set()
    for _, output in errors:
        for m in re.finditer(r"ModuleNotFoundError: No module named '([^']+)'", output):
            packages.add(m.group(1))
    return packages


def extract_missing_files(errors: list) -> set:
    """Собирает все отсутствующие файлы из списка ошибок."""
    files = set()
    for _, output in errors:
        for m in re.finditer(
            r"FileNotFoundError:.*?No such file or directory: '([^']+)'", output
        ):
            files.add(m.group(1))
    return files


# ============================================================================
# Поиск и запуск
# ============================================================================

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


# ============================================================================
# Итоговая сводка
# ============================================================================

def write_summary(log, ok_list, errors, total):
    """Пишет итоговую сводку в лог и возвращает строки для консоли."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("ИТОГОВАЯ СВОДКА")
    lines.append("=" * 70)
    lines.append(f"Всего файлов: {total}")
    lines.append(f"Успешно:      {len(ok_list)}")
    lines.append(f"Ошибок:       {len(errors)}")
    lines.append("=" * 70)

    if errors:
        # Группировка по категориям
        categories = defaultdict(list)
        for script_path, output in errors:
            cat = classify_error(output, script_path)
            categories[cat].append(script_path.relative_to(IOPT_ROOT))

        lines.append("\nРАЗБОР ОШИБОК ПО КАТЕГОРИЯМ:")
        lines.append("-" * 70)

        # Сортируем категории по количеству файлов (больше -> выше)
        for cat in sorted(categories, key=lambda c: -len(categories[c])):
            files = categories[cat]
            lines.append(f"\n[{len(files)}] {cat}")
            for f in files:
                lines.append(f"      - {f}")

        # Секция «что установить»
        missing_pkgs = extract_missing_packages(errors)
        if missing_pkgs:
            lines.append("\n" + "-" * 70)
            lines.append("ОТСУТСТВУЮЩИЕ ПАКЕТЫ (установить через pip):")
            lines.append("-" * 70)
            for pkg in sorted(missing_pkgs):
                lines.append(f"  pip install {pkg}")

        # Секция «что скачать»
        missing_files = extract_missing_files(errors)
        if missing_files:
            lines.append("\n" + "-" * 70)
            lines.append("ОТСУТСТВУЮЩИЕ ФАЙЛЫ ДАННЫХ (скачать вручную):")
            lines.append("-" * 70)
            for f in sorted(missing_files):
                lines.append(f"  {f}")

        # Секция «реальные баги в коде»
        real_bugs = [
            (p, o) for p, o in errors
            if classify_error(o, p) not in (
                *[f"Missing package: {p}" for p in missing_pkgs],
                *[f"Missing file: {Path(f).name}" for f in missing_files],
                "Timeout",
                "No tests ran (unittest)",
            )
        ]
        if real_bugs:
            lines.append("\n" + "-" * 70)
            lines.append("РЕАЛЬНЫЕ БАГИ В КОДЕ iOpt (требуют правки):")
            lines.append("-" * 70)
            for p, o in real_bugs:
                cat = classify_error(o, p)
                lines.append(f"  - {p.relative_to(IOPT_ROOT)}")
                lines.append(f"      {cat}")

    lines.append("\n" + "=" * 70)
    return lines


# ============================================================================
# main
# ============================================================================

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

    ok_list = []
    errors = []

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write(f"# Результаты запуска примеров iOpt\n")
        log.write(f"# Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"# Всего файлов: {len(py_files)}\n")
        log.write("=" * 70 + "\n\n")

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
                ok_list.append(script_path)
                print(f"[V]OK ({elapsed:.2f} сек)")
                log.write("Результат: OK\n\n")
            else:
                errors.append((script_path, output))
                print(f"[X]ERROR ({elapsed:.2f} сек)")
                log.write("Результат: ERROR\n")
                log.write("--- ЛОГ ОШИБКИ ---\n")
                log.write(output + "\n\n")

            log.flush()

        # === ИТОГОВАЯ СВОДКА ===
        summary_lines = write_summary(log, ok_list, errors, len(py_files))
        for line in summary_lines:
            log.write(line + "\n")
            print(line)

    print(f"\nЛог записан в: {LOG_FILE}")


if __name__ == "__main__":
    main()