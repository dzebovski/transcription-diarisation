"""
transcribe.py — транскрипція аудіо через WhisperX.
Включає alignment для точних таймкодів.
"""

import argparse
import json
import sys
from pathlib import Path

import whisperx


def transcribe(
    input_path: str,
    output_dir: str = "output",
    model_name: str = "large-v3",
    language: str = "uk",
    batch_size: int = 16,
    compute_type: str = "float16",
    device: str = "cuda",
) -> dict:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        print(f"[ERROR] Файл не знайдено: {input_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Завантаження моделі
    print(f"[INFO] Завантажую модель WhisperX: {model_name} ({compute_type})")
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    # 2. Транскрипція
    print(f"[INFO] Транскрибую: {input_path}")
    audio = whisperx.load_audio(str(input_path))
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    print(f"[INFO] Отримано {len(result['segments'])} сегментів")

    # 3. Alignment (точні таймкоди на рівні слів)
    print("[INFO] Виконую alignment...")
    model_a, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # 4. Збереження результату
    output_json = output_dir / f"{input_path.stem}_transcription.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] Транскрипцію збережено: {output_json}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Транскрипція аудіо через WhisperX")
    parser.add_argument("input", help="Шлях до WAV файлу")
    parser.add_argument("-o", "--output-dir", default="output", help="Папка для результатів")
    parser.add_argument("--model", default="large-v3", help="Модель Whisper")
    parser.add_argument("--language", default="uk", help="Мова аудіо (uk, en, ...)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    transcribe(
        args.input,
        args.output_dir,
        args.model,
        args.language,
        args.batch_size,
        args.compute_type,
        args.device,
    )


if __name__ == "__main__":
    main()
