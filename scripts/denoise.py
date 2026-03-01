"""
denoise.py — ізоляція голосу від фонових шумів через Demucs.
Вхід: WAV файл. Вихід: WAV тільки з голосом (vocals stem).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def denoise(
    input_path: str,
    output_path: str,
    model: str = "htdemucs",
    stem: str = "vocals",
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"[ERROR] Файл не знайдено: {input_path}")
        sys.exit(1)

    # Demucs зберігає результат у: separated/<model>/<filename>/<stem>.wav
    separated_dir = Path("separated")
    separated_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "demucs",
        "--two-stems", stem,          # витягуємо тільки vocals
        "-n", model,
        "--out", str(separated_dir),
        str(input_path),
    ]

    print(f"[INFO] Запускаю Demucs (модель: {model}, stem: {stem})")
    print(f"[INFO] Вхід: {input_path}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("[ERROR] Demucs завершився з помилкою.")
        sys.exit(1)

    # Знаходимо вихідний файл
    stem_file = separated_dir / model / input_path.stem / f"{stem}.wav"

    if not stem_file.exists():
        print(f"[ERROR] Demucs не створив файл: {stem_file}")
        sys.exit(1)

    # Копіюємо у output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stem_file, output_path)

    print(f"[OK] Ізольований голос збережено: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Ізоляція голосу через Demucs")
    parser.add_argument("input", help="Шлях до WAV файлу")
    parser.add_argument(
        "-o", "--output",
        help="Шлях до вихідного WAV (за замовчуванням: output/<назва>_vocals.wav)"
    )
    parser.add_argument(
        "--model", default="htdemucs",
        help="Модель Demucs (htdemucs або mdx_extra)"
    )
    parser.add_argument(
        "--stem", default="vocals",
        help="Stem для витягування (за замовчуванням: vocals)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = args.output or f"output/{input_path.stem}_vocals.wav"

    denoise(args.input, output_path, args.model, args.stem)


if __name__ == "__main__":
    main()
