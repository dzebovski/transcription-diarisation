"""
pipeline.py — main CLI entry point.

Automatically resolves all paths from episode name.

Usage:
  python scripts/pipeline.py episode_01
  python scripts/pipeline.py episode_01 --no-denoise
  python scripts/pipeline.py episode_01 --min-speakers 2 --max-speakers 4

Expected input structure:
  input/episode_01/episode_01.mp4
  input/episode_01/episode_01.speakers.txt   (optional)

Output structure:
  output/episode_01/episode_01.wav
  output/episode_01/episode_01_vocals.wav
  output/episode_01/episode_01_diarized.txt
  output/episode_01/episode_01.srt
  output/episode_01/episode_01_diarized.json
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

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


def resolve_episode_paths(episode: str) -> dict:
    """
    Resolves all input/output paths from episode name.
    Returns dict with all paths needed for the pipeline.
    """
    ep = episode.strip("/\\")
    input_dir = Path("input") / ep
    output_dir = Path("output") / ep

    # Find video file in input dir
    video_path = None
    for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        candidate = input_dir / f"{ep}{ext}"
        if candidate.exists():
            video_path = candidate
            break

    # If not found by episode name, take first video file in dir
    if video_path is None and input_dir.exists():
        for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
            candidates = list(input_dir.glob(f"*{ext}"))
            if candidates:
                video_path = candidates[0]
                break

    speakers_file = input_dir / f"{ep}.speakers.txt"

    return {
        "episode": ep,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "video_path": video_path,
        "speakers_file": speakers_file if speakers_file.exists() else None,
        "raw_audio": output_dir / f"{ep}.wav",
        "clean_audio": output_dir / f"{ep}_vocals.wav",
        "transcription_json": output_dir / f"{ep}_vocals_transcription.json",
    }


def check_environment():
    """Checks that all required modules are importable."""
    print("[CHECK] Verifying environment...")
    errors = []

    checks = [
        ("torch", "PyTorch"),
        ("torchaudio", "torchaudio"),
        ("whisperx", "WhisperX"),
        ("pyannote.audio", "pyannote.audio"),
        ("demucs", "Demucs"),
        ("dotenv", "python-dotenv"),
        ("yaml", "PyYAML"),
        ("pandas", "pandas"),
        ("soundfile", "soundfile"),
    ]

    for module, name in checks:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            errors.append(name)

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA ({torch.version.cuda}) — {torch.cuda.get_device_name(0)}")
        else:
            print("  [WARN] CUDA not available — will run on CPU (slow)")
    except Exception:
        pass

    # Check HF token
    import os
    if os.getenv("HF_TOKEN"):
        print("  [OK] HF_TOKEN found in .env")
    else:
        print("  [WARN] HF_TOKEN not set — diarization will fail")

    if errors:
        print(f"\n[ERROR] Missing modules: {', '.join(errors)}")
        print("Run: pip install " + " ".join(e.lower().replace(" ", "-") for e in errors))
        sys.exit(1)

    print("[CHECK] Environment OK\n")


def run_pipeline(
    episode: str,
    config: dict,
    no_denoise: bool = False,
    min_speakers: int = None,
    max_speakers: int = None,
    skip_check: bool = False,
):
    if not skip_check:
        check_environment()

    start_time = time.time()
    paths = resolve_episode_paths(episode)

    # Validate input
    if paths["video_path"] is None:
        print(f"[ERROR] No video file found in: {paths['input_dir']}")
        print(f"  Expected: input/{episode}/{episode}.mp4")
        sys.exit(1)

    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Episode:  {paths['episode']}")
    print(f"  Video:    {paths['video_path']}")
    print(f"  Speakers: {paths['speakers_file'] or 'not provided (will use SPEAKER_XX labels)'}")
    print(f"  Output:   {paths['output_dir']}/")
    print(f"{'='*60}\n")

    # === STEP 1: Extract audio ===
    print("[STEP 1/4] Extracting audio from video...")
    audio_cfg = config.get("audio", {})
    extract_audio(
        str(paths["video_path"]),
        str(paths["raw_audio"]),
        sample_rate=audio_cfg.get("sample_rate", 16000),
    )

    # === STEP 2: Voice isolation (optional) ===
    denoise_cfg = config.get("denoise", {})
    use_denoise = not no_denoise and denoise_cfg.get("enabled", True)

    if use_denoise:
        print("\n[STEP 2/4] Isolating voice with Demucs...")
        denoise(
            str(paths["raw_audio"]),
            str(paths["clean_audio"]),
            model=denoise_cfg.get("model", "htdemucs"),
            stem=denoise_cfg.get("stem", "vocals"),
        )
        audio_for_transcription = paths["clean_audio"]
    else:
        print("\n[STEP 2/4] Skipping denoising (--no-denoise)")
        audio_for_transcription = paths["raw_audio"]

    # === STEP 3: Transcription ===
    print("\n[STEP 3/4] Transcribing with WhisperX...")
    whisper_cfg = config.get("whisper", {})
    transcription = transcribe(
        str(audio_for_transcription),
        output_dir=str(paths["output_dir"]),
        model_name=whisper_cfg.get("model", "large-v3"),
        language=whisper_cfg.get("language", "uk"),
        batch_size=whisper_cfg.get("batch_size", 16),
        compute_type=whisper_cfg.get("compute_type", "float16"),
        device=whisper_cfg.get("device", "cuda"),
    )

    # === STEP 4: Diarization ===
    print("\n[STEP 4/4] Diarizing speakers...")
    diar_cfg = config.get("diarization", {})
    diarize(
        str(audio_for_transcription),
        transcription,
        output_dir=str(paths["output_dir"]),
        min_speakers=min_speakers or diar_cfg.get("min_speakers", 4),
        max_speakers=max_speakers or diar_cfg.get("max_speakers", 6),
        device=whisper_cfg.get("device", "cuda"),
        speakers_file=str(paths["speakers_file"]) if paths["speakers_file"] else None,
    )

    # === DONE ===
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*60}")
    print(f"  Done! Processing time: {minutes}m {seconds}s")
    print(f"  Results in: {paths['output_dir']}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Transcription + diarization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pipeline.py episode_01
  python scripts/pipeline.py episode_01 --no-denoise
  python scripts/pipeline.py episode_01 --min-speakers 2 --max-speakers 3
  python scripts/pipeline.py episode_01 --check-only

Input expected at:
  input/episode_01/episode_01.mp4
  input/episode_01/episode_01.speakers.txt
        """,
    )
    parser.add_argument("episode", help="Episode name (e.g. episode_01)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-denoise", action="store_true", help="Skip Demucs voice isolation")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--check-only", action="store_true", help="Only check environment, don't run pipeline")
    parser.add_argument("--skip-check", action="store_true", help="Skip environment check")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.check_only:
        check_environment()
        return

    run_pipeline(
        args.episode,
        config,
        no_denoise=args.no_denoise,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        skip_check=args.skip_check,
    )


if __name__ == "__main__":
    main()
