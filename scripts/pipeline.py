"""
pipeline.py — головний CLI entry point.
Запускає повний пайплайн: відео → аудіо → денойз → транскрипція → діаризація.

Використання:
  python scripts/pipeline.py input/video.mp4
  python scripts/pipeline.py input/video.mp4 --no-denoise
  python scripts/pipeline.py input/video.mp4 --min-speakers 2 --max-speakers 4
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Підключаємо наші модулі
sys.path.insert(0, str(Path(__file__).parent))
from extract_audio import extract_audio
from denoise import denoise
from transcribe import transcribe
from diarize import diarize


def load_config(config_path: str = "config.yaml") -> dict:
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def run_pipeline(
    input_video: str,
    config: dict,
    no_denoise: bool = False,
    min_speakers: int = None,
    max_speakers: int = None,
):
    start_time = time.time()
    input_path = Path(input_video)

    if not input_path.exists():
        print(f"[ERROR] Файл не знайдено: {input_path}")
        sys.exit(1)

    stem = input_path.stem
    output_dir = Path(config.get("output", {}).get("directory", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Пайплайн транскрипції: {input_path.name}")
    print(f"{'='*60}\n")

    # === КРОК 1: Вилучення аудіо ===
    print("[КРОК 1/4] Вилучення аудіо з відео...")
    audio_cfg = config.get("audio", {})
    raw_audio = output_dir / f"{stem}.wav"
    extract_audio(
        str(input_path),
        str(raw_audio),
        sample_rate=audio_cfg.get("sample_rate", 16000),
    )

    # === КРОК 2: Денойзинг (опціонально) ===
    denoise_cfg = config.get("denoise", {})
    use_denoise = not no_denoise and denoise_cfg.get("enabled", True)

    if use_denoise:
        print("\n[КРОК 2/4] Ізоляція голосу через Demucs...")
        clean_audio = output_dir / f"{stem}_vocals.wav"
        denoise(
            str(raw_audio),
            str(clean_audio),
            model=denoise_cfg.get("model", "htdemucs"),
            stem=denoise_cfg.get("stem", "vocals"),
        )
        audio_for_transcription = clean_audio
    else:
        print("\n[КРОК 2/4] Денойзинг пропущено (--no-denoise)")
        audio_for_transcription = raw_audio

    # === КРОК 3: Транскрипція ===
    print("\n[КРОК 3/4] Транскрипція через WhisperX...")
    whisper_cfg = config.get("whisper", {})
    transcription = transcribe(
        str(audio_for_transcription),
        output_dir=str(output_dir),
        model_name=whisper_cfg.get("model", "large-v3"),
        language=whisper_cfg.get("language", "uk"),
        batch_size=whisper_cfg.get("batch_size", 16),
        compute_type=whisper_cfg.get("compute_type", "float16"),
        device=whisper_cfg.get("device", "cuda"),
    )

    # === КРОК 4: Діаризація ===
    print("\n[КРОК 4/4] Діаризація (розділення спікерів)...")
    diar_cfg = config.get("diarization", {})
    diarize(
        str(audio_for_transcription),
        transcription,
        output_dir=str(output_dir),
        min_speakers=min_speakers or diar_cfg.get("min_speakers", 4),
        max_speakers=max_speakers or diar_cfg.get("max_speakers", 6),
        device=whisper_cfg.get("device", "cuda"),
    )

    # === ГОТОВО ===
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*60}")
    print(f"  Готово! Час обробки: {minutes}хв {seconds}с")
    print(f"  Результати у папці: {output_dir}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Пайплайн транскрипції + діаризації відео",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  python scripts/pipeline.py input/lecture.mp4
  python scripts/pipeline.py input/lecture.mp4 --no-denoise
  python scripts/pipeline.py input/lecture.mp4 --min-speakers 2 --max-speakers 3
        """,
    )
    parser.add_argument("input", help="Шлях до відео файлу")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Шлях до конфіг файлу (за замовчуванням: config.yaml)"
    )
    parser.add_argument(
        "--no-denoise", action="store_true",
        help="Пропустити крок очистки голосу через Demucs"
    )
    parser.add_argument("--min-speakers", type=int, help="Мінімальна кількість спікерів")
    parser.add_argument("--max-speakers", type=int, help="Максимальна кількість спікерів")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(
        args.input,
        config,
        no_denoise=args.no_denoise,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )


if __name__ == "__main__":
    main()
