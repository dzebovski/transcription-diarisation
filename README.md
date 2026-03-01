# transcription-diarisation

Local AI pipeline for transcription and speaker diarization of video content. Fully offline — no cloud services, no data leaves your machine.

## What it does

Takes a video file (e.g. a 1-hour TV show in Ukrainian), extracts audio, isolates voices from background noise, transcribes speech with word-level timestamps, and splits the transcript by speaker. Output is a clean, readable text file with named speakers and timecodes.

**Example output:**
```
[00:00 - 00:07] Host Dzidzio:
  You are watching "Guess the Melody" — an original show on Svit-TV!

[01:09 - 01:46] Eugen Klopotenko:
  50% will go to Serhiy Prytula fund, and the other 50% to my charity foundation...
```

## Hardware

- CPU: Ryzen 7
- GPU: RTX 4080 (16GB VRAM, CUDA 12.x)
- All processing is local

## Stack

| Step | Tool | Notes |
|------|------|-------|
| Audio extraction | `ffmpeg` | 16kHz, mono, WAV |
| Voice isolation | `Demucs` (htdemucs) | Removes background noise/music |
| Transcription | `WhisperX` + `faster-whisper` | Model: `large-v3`, Ukrainian |
| Word alignment | `WhisperX` align | Word-level timestamps |
| Diarization | `pyannote.audio 4.x` | Speaker separation |
| Post-processing | Merge + speaker mapping | Named speakers from file |

## Pipeline

```
video.mp4
  └─ extract_audio.py   →  audio.wav
       └─ denoise.py    →  audio_vocals.wav
            └─ transcribe.py  →  transcription.json
                 └─ diarize.py →  diarized.txt / .srt / .json
```

## Usage

```bash
conda activate transcription

# Full pipeline — just pass the episode name:
python scripts/pipeline.py episode_01

# With options:
python scripts/pipeline.py episode_01 --no-denoise
python scripts/pipeline.py episode_01 --min-speakers 2 --max-speakers 3

# Cleanup when done:
python scripts/cleanup.py episode_01
```

Input expected at `input/episode_01/episode_01.mp4` and `input/episode_01/episode_01.speakers.txt`.
See `HOW_TO.md` for full step-by-step instructions.

## Configuration

- `.env` — secrets (HF Token for pyannote)
- `config.yaml` — pipeline parameters (model, speakers count, language, etc.)

## Output files

| File | Description |
|------|-------------|
| `*_diarized.txt` | Human-readable transcript with speaker names and timecodes |
| `*.srt` | Subtitle file |
| `*_diarized.json` | Full JSON with raw and merged segments |
| `*_transcription.json` | WhisperX output with word-level alignment |

## Setup

```bash
conda env create -f environment.yml
conda activate transcription
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install whisperx pyannote.audio demucs soundfile hf_xet
cp .env.example .env  # add your HF_TOKEN
```
