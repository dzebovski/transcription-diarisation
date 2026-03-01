"""
pipeline.py — main CLI entry point.

Аргумент — папка в input/, з якою працюємо (наприклад episode_01 або ep01).
У папці має бути рівно один відеофайл (.mp4 тощо) та опційно рівно один
файл *.speakers.txt з іменами спікерів; якщо speakers немає — використовуються
SPEAKER_01, SPEAKER_02, …

Usage:
  python scripts/pipeline.py episode_01
  python scripts/pipeline.py episode_01 --no-denoise
  python scripts/pipeline.py episode_01 --min-speakers 2 --max-speakers 4

Expected input structure:
  input/<episode>/   — одна папка (ім'я вказує користувач)
    один відеофайл   — .mp4, .mkv, .avi, .mov або .webm (користувач кладе сам)
    один *.speakers.txt (optional) — якщо немає, дефолт: SPEAKER_01, SPEAKER_02, …

Output structure:
  output/<episode>/<episode>.wav
  output/<episode>/<episode>_vocals.wav
  output/<episode>/<episode>_diarized.txt
  output/<episode>/<episode>.srt
  output/<episode>/<episode>_diarized.json
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

# ANSI green for progress/success (Windows 10+ and Unix support these)
_GREEN = "\033[92m"
_RESET = "\033[0m"
_DOT = "\u2022"   # bullet •
_CHECK = "\u2713"  # checkmark ✓


def _ok(msg: str) -> str:
    """Wrap message in green for success/progress (only if stdout is a TTY)."""
    if sys.stdout.isatty():
        return f"{_GREEN}{msg}{_RESET}"
    return msg


def _step_start(step: str, total: int, description: str) -> None:
    print(_ok(f"  {_DOT} [STEP {step}/{total}] {description}"))
    sys.stdout.flush()


def _step_done(description: str) -> None:
    print(_ok(f"    {_CHECK} {description}"))
    sys.stdout.flush()
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
    Resolves all input/output paths from episode name (folder under input/).
    Returns dict with all paths needed for the pipeline.

    Requires exactly one video file in input_dir; optionally exactly one
    *.speakers.txt. Exits with error if multiple videos or multiple
    *.speakers.txt are found.
    """
    ep = episode.strip("/\\")
    input_dir = Path("input") / ep
    output_dir = Path("output") / ep

    video_extensions = [".mp4", ".mkv", ".avi", ".mov", ".webm"]
    video_candidates = []
    if input_dir.exists():
        for ext in video_extensions:
            video_candidates.extend(input_dir.glob(f"*{ext}"))

    if len(video_candidates) == 0:
        video_path = None
    elif len(video_candidates) > 1:
        print(f"[ERROR] У папці має бути лише один відеофайл: {input_dir}")
        print(f"  Знайдено: {[p.name for p in video_candidates]}")
        sys.exit(1)
    else:
        video_path = video_candidates[0]

    speakers_candidates = list(input_dir.glob("*.speakers.txt")) if input_dir.exists() else []
    if len(speakers_candidates) > 1:
        print(f"[ERROR] У папці має бути лише один файл *.speakers.txt: {input_dir}")
        print(f"  Знайдено: {[p.name for p in speakers_candidates]}")
        sys.exit(1)
    speakers_file = speakers_candidates[0] if len(speakers_candidates) == 1 else None

    return {
        "episode": ep,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "video_path": video_path,
        "speakers_file": speakers_file,
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
            print(_ok(f"  [OK] {name}"))
        except ImportError:
            print(f"  [MISSING] {name}")
            errors.append(name)

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(_ok(f"  [OK] CUDA ({torch.version.cuda}) — {torch.cuda.get_device_name(0)}"))
        else:
            print("  [WARN] CUDA not available — will run on CPU (slow)")
    except Exception:
        pass

    # Check HF token
    import os
    if os.getenv("HF_TOKEN"):
        print(_ok("  [OK] HF_TOKEN found in .env"))
    else:
        print("  [WARN] HF_TOKEN not set — diarization will fail")

    if errors:
        print(f"\n[ERROR] Missing modules: {', '.join(errors)}")
        print("Run: pip install " + " ".join(e.lower().replace(" ", "-") for e in errors))
        sys.exit(1)

    print(_ok("[CHECK] Environment OK\n"))


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
        print(f"[ERROR] В папці немає відеофайлу: {paths['input_dir']}")
        print(f"  Покладіть один файл .mp4 (або .mkv/.avi/.mov/.webm) у цю папку.")
        sys.exit(1)

    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Episode:  {paths['episode']}")
    print(f"  Video:    {paths['video_path']}")
    print(f"  Speakers: {paths['speakers_file'] or 'не вказано (буде SPEAKER_01, SPEAKER_02, …)'}")
    print(f"  Output:   {paths['output_dir']}/")
    print(f"{'='*60}\n")

    # === STEP 1: Extract audio ===
    _step_start(1, 4, "Extracting audio from video...")
    audio_cfg = config.get("audio", {})
    extract_audio(
        str(paths["video_path"]),
        str(paths["raw_audio"]),
        sample_rate=audio_cfg.get("sample_rate", 16000),
    )
    _step_done("Audio extracted.")

    # === STEP 2: Voice isolation (optional) ===
    denoise_cfg = config.get("denoise", {})
    use_denoise = not no_denoise and denoise_cfg.get("enabled", True)

    if use_denoise:
        _step_start(2, 4, "Isolating voice with Demucs...")
        denoise(
            str(paths["raw_audio"]),
            str(paths["clean_audio"]),
            model=denoise_cfg.get("model", "htdemucs"),
            stem=denoise_cfg.get("stem", "vocals"),
        )
        _step_done("Voice isolated.")
        audio_for_transcription = paths["clean_audio"]
    else:
        _step_start(2, 4, "Skipping denoising (--no-denoise)")
        _step_done("Skipped.")
        audio_for_transcription = paths["raw_audio"]

    # === STEP 3: Transcription ===
    _step_start(3, 4, "Transcribing with WhisperX...")
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
    _step_done("Transcription done.")

    # === STEP 4: Diarization ===
    _step_start(4, 4, "Diarizing speakers...")
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
    _step_done("Diarization done.")

    # === DONE ===
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print()
    print(_ok(f"{'='*60}"))
    print(_ok(f"  Done! Processing time: {minutes}m {seconds}s"))
    print(_ok(f"  Results in: {paths['output_dir']}/"))
    print(_ok(f"{'='*60}\n"))


def main():
    parser = argparse.ArgumentParser(
        description="Transcription + diarization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pipeline.py episode_01
  python scripts/pipeline.py ep01 --no-denoise
  python scripts/pipeline.py episode_01 --min-speakers 2 --max-speakers 3
  python scripts/pipeline.py --check-only   # no episode name needed

Input: папка в input/ (наприклад input/episode_01/ або input/ep01/).
  У папці — рівно один відеофайл (.mp4 тощо) і опційно рівно один *.speakers.txt.
  Якщо speakers.txt немає — будуть SPEAKER_01, SPEAKER_02, …
        """,
    )
    parser.add_argument(
        "episode",
        nargs="?",
        default=None,
        help="Episode name (e.g. episode_01); optional when using --check-only",
    )
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

    if not args.episode:
        parser.error("episode name is required (e.g. episode_01)")

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
