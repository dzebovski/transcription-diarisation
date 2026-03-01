"""
diarize.py — діаризація (розділення спікерів) через pyannote.audio + WhisperX.
Поєднує результат транскрипції з мітками спікерів.
Підтримує: злиття блоків одного спікера, мапінг імен зі speakers файлу.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import whisperx
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()


def load_speaker_names(speakers_file: str) -> dict:
    """
    Завантажує список імен спікерів з файлу (один рядок = один спікер).
    Повертає dict: {'SPEAKER_00': 'Host Dzidzio', 'SPEAKER_01': 'Eugen Klopotenko', ...}
    Порядок: ведучий зазвичай говорить першим → SPEAKER_00.
    """
    speakers_path = Path(speakers_file)
    if not speakers_path.exists():
        return {}

    names = []
    with open(speakers_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)

    return {f"SPEAKER_{i:02d}": name for i, name in enumerate(names)}


def merge_segments(segments: list, speaker_map: dict, gap_threshold: float = 1.5) -> list:
    """
    Зливає сусідні сегменти одного спікера в один логічний блок.
    gap_threshold: максимальна пауза між сегментами (сек) для злиття.
    """
    if not segments:
        return []

    merged = []
    current = None

    for seg in segments:
        raw_speaker = seg.get("speaker", "UNKNOWN")
        speaker = speaker_map.get(raw_speaker, raw_speaker)
        text = seg.get("text", "").strip()
        start = seg["start"]
        end = seg["end"]

        if current is None:
            current = {"start": start, "end": end, "speaker": speaker, "text": text}
        elif (
            speaker == current["speaker"]
            and (start - current["end"]) <= gap_threshold
        ):
            # Зливаємо з поточним блоком
            current["end"] = end
            current["text"] = current["text"].rstrip() + " " + text
        else:
            merged.append(current)
            current = {"start": start, "end": end, "speaker": speaker, "text": text}

    if current:
        merged.append(current)

    return merged


def diarize(
    audio_path: str,
    transcription: dict,
    output_dir: str = "output",
    min_speakers: int = 4,
    max_speakers: int = 6,
    device: str = "cuda",
    speakers_file: str = None,
    gap_threshold: float = 1.5,
) -> dict:
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN не знайдено. Додай у .env файл.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Завантажуємо імена спікерів якщо є
    speaker_map = {}
    if speakers_file:
        speaker_map = load_speaker_names(speakers_file)
        if speaker_map:
            print(f"[INFO] Завантажено спікерів: {list(speaker_map.values())}")

    # 1. Завантажуємо модель діаризації через pyannote напряму
    print(f"[INFO] Завантажую pyannote модель (min={min_speakers}, max={max_speakers})")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    import torch
    pipeline.to(torch.device(device))

    # 2. Запускаємо діаризацію
    # Передаємо аудіо як waveform (обхід проблеми з torchcodec на Windows)
    print(f"[INFO] Завантажую аудіо: {audio_path}")
    import torch
    import torchaudio
    waveform, sample_rate = torchaudio.load(str(audio_path))
    audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

    print(f"[INFO] Діаризую: {audio_path}")
    diarization = pipeline(
        audio_in_memory,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # 3. Конвертуємо результат pyannote у формат для whisperx
    import pandas as pd

    annotation = diarization.speaker_diarization
    diarize_segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        diarize_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    print(f"[INFO] Знайдено {len(diarize_segments)} діаризованих сегментів")

    diarize_df = pd.DataFrame(diarize_segments)
    diarize_df["label"] = diarize_df["speaker"]

    # 4. Присвоюємо мітки спікерів до сегментів транскрипції
    print("[INFO] Призначаю спікерів до сегментів...")
    result = whisperx.assign_word_speakers(diarize_df, transcription)

    # 4. Зливаємо сегменти в логічні блоки
    print(f"[INFO] Зливаю сегменти (поріг паузи: {gap_threshold}с)...")
    merged = merge_segments(result["segments"], speaker_map, gap_threshold)
    print(f"[INFO] Сегментів до злиття: {len(result['segments'])}, після: {len(merged)}")

    # 5. Збереження JSON (оригінал + merged)
    stem = Path(audio_path).stem.replace("_vocals", "")
    output_json = output_dir / f"{stem}_diarized.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"raw": result, "merged": merged}, f, ensure_ascii=False, indent=2)

    # 6. Зберігаємо читабельний TXT
    output_txt = output_dir / f"{stem}_diarized.txt"
    _save_txt(merged, output_txt)

    # 7. Зберігаємо SRT
    output_srt = output_dir / f"{stem}.srt"
    _save_srt(merged, output_srt)

    print(f"[OK] Збережено:")
    print(f"     JSON: {output_json}")
    print(f"     TXT:  {output_txt}")
    print(f"     SRT:  {output_srt}")

    return result


def _format_time(seconds: float) -> str:
    """Форматує секунди у HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_srt_time(seconds: float) -> str:
    """Форматує секунди у SRT формат HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _save_txt(segments: list, output_path: Path):
    """Зберігає транскрипт у читабельний текстовий формат."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            speaker = seg["speaker"]
            start = _format_time(seg["start"])
            end = _format_time(seg["end"])
            text = seg["text"].strip()

            f.write(f"\n[{start} - {end}] {speaker}:\n")
            f.write(f"  {text}\n")


def _save_srt(segments: list, output_path: Path):
    """Зберігає транскрипт у SRT формат."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = _format_srt_time(seg["start"])
            end = _format_srt_time(seg["end"])
            speaker = seg["speaker"]
            text = seg["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"[{speaker}] {text}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Діаризація через pyannote + WhisperX")
    parser.add_argument("audio", help="Шлях до WAV файлу")
    parser.add_argument("transcription", help="Шлях до JSON файлу транскрипції")
    parser.add_argument("-o", "--output-dir", default="output")
    parser.add_argument("--min-speakers", type=int, default=4)
    parser.add_argument("--max-speakers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--speakers",
        help="Файл зі списком імен спікерів (один рядок = один спікер)",
        default=None,
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=1.5,
        help="Максимальна пауза між репліками для злиття (сек, за замовч: 1.5)",
    )
    args = parser.parse_args()

    with open(args.transcription, "r", encoding="utf-8") as f:
        transcription = json.load(f)

    diarize(
        args.audio,
        transcription,
        args.output_dir,
        args.min_speakers,
        args.max_speakers,
        args.device,
        speakers_file=args.speakers,
        gap_threshold=args.gap,
    )


if __name__ == "__main__":
    main()
