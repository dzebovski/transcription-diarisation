"""
extract_audio.py — витягує аудіо з відео через ffmpeg.
Вихід: 16kHz, mono, WAV.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def extract_audio(input_path: str, output_path: str, sample_rate: int = 16000) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"[ERROR] Файл не знайдено: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                        # перезаписати якщо існує
        "-i", str(input_path),
        "-vn",                       # без відео
        "-ac", "1",                  # mono
        "-ar", str(sample_rate),     # sample rate
        "-acodec", "pcm_s16le",      # WAV 16-bit
        str(output_path),
    ]

    print(f"[INFO] Витягую аудіо з: {input_path}")
    print(f"[INFO] Збереження в:    {output_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] ffmpeg помилка:\n{result.stderr}")
        sys.exit(1)

    print(f"[OK] Аудіо збережено: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Витягти аудіо з відео файлу")
    parser.add_argument("input", help="Шлях до відео файлу (.mp4, .mkv, тощо)")
    parser.add_argument(
        "-o", "--output",
        help="Шлях до вихідного WAV файлу (за замовчуванням: output/<назва>.wav)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Sample rate (за замовчуванням: 16000)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = args.output or f"output/{input_path.stem}.wav"

    extract_audio(args.input, output_path, args.sample_rate)


if __name__ == "__main__":
    main()
