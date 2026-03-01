# HOW TO: Run the Transcription Pipeline

## Prerequisites

Before running for a new episode, make sure:
1. You are in the project directory: `C:\work\transcription-diarisation`
2. Conda environment is activated
3. `.env` file exists with your HF Token

---

## Step 0 — Activate environment & verify

Every session starts here:

```bash
conda activate transcription
```

Check that everything is installed correctly:

```bash
python scripts/pipeline.py --check-only placeholder
```

Expected output — all green OKs:
```
[CHECK] Verifying environment...
  [OK] PyTorch
  [OK] torchaudio
  [OK] WhisperX
  [OK] pyannote.audio
  [OK] Demucs
  [OK] python-dotenv
  [OK] PyYAML
  [OK] pandas
  [OK] soundfile
  [OK] CUDA (12.8) — NVIDIA GeForce RTX 4080
  [OK] HF_TOKEN found in .env
[CHECK] Environment OK
```

---

## Step 1 — Prepare input files

Create a folder for the episode and place the video and speakers list inside:

```
input/
  episode_02/
    episode_02.mp4
    episode_02.speakers.txt
```

**speakers.txt** — one name per line, in order of first appearance (who speaks first = line 1):

```
Host Dzidzio
Eugen Klopotenko
Tina Karol
Pivovarov
```

> If you are unsure of the order, use generic names like Guest1, Guest2.
> You can rename speakers manually in the output .txt file afterwards.

---

## Step 2 — Run the full pipeline

```bash
python scripts/pipeline.py episode_02
```

This single command runs all 4 steps automatically:

| Step | What happens | Output file |
|------|-------------|-------------|
| 1/4 | Extracts audio from video (ffmpeg) | `output/episode_02/episode_02.wav` |
| 2/4 | Isolates voice from music/noise (Demucs) | `output/episode_02/episode_02_vocals.wav` |
| 3/4 | Transcribes speech (WhisperX large-v3) | `output/episode_02/episode_02_vocals_transcription.json` |
| 4/4 | Splits by speaker (pyannote) | `output/episode_02/episode_02_diarized.txt` + `.srt` + `.json` |

---

## Step 3 — Check the output

Your results are in `output/episode_02/`:

```
output/
  episode_02/
    episode_02.wav                          # raw extracted audio
    episode_02_vocals.wav                   # denoised voice only
    episode_02_vocals_transcription.json    # word-level transcript
    episode_02_diarized.txt                 # MAIN RESULT — readable transcript
    episode_02.srt                          # subtitle file
    episode_02_diarized.json                # full JSON (raw + merged segments)
```

Open `episode_02_diarized.txt` — it looks like this:

```
[00:00 - 00:07] Host Dzidzio:
  You are watching Guess the Melody, an original show on Svit-TV!

[01:09 - 01:46] Eugen Klopotenko:
  50% will go to Serhiy Prytula fund, and the other 50% to my foundation...

[01:46 - 01:49] Host Dzidzio:
  A very noble mission.
```

---

## Options & flags

```bash
# Skip voice isolation (faster, use if audio is already clean)
python scripts/pipeline.py episode_02 --no-denoise

# Override speaker count (if diarization splits incorrectly)
python scripts/pipeline.py episode_02 --min-speakers 2 --max-speakers 3

# Skip environment check on re-runs
python scripts/pipeline.py episode_02 --skip-check
```

---

## Run steps individually

If the full pipeline fails midway, you can re-run from any specific step:

```bash
# Step 1: Extract audio only
python scripts/extract_audio.py input/episode_02/episode_02.mp4 -o output/episode_02/episode_02.wav

# Step 2: Denoise only
python scripts/denoise.py output/episode_02/episode_02.wav -o output/episode_02/episode_02_vocals.wav

# Step 3: Transcribe only
python scripts/transcribe.py output/episode_02/episode_02_vocals.wav -o output/episode_02/

# Step 4: Diarize only
python scripts/diarize.py output/episode_02/episode_02_vocals.wav output/episode_02/episode_02_vocals_transcription.json \
  --speakers input/episode_02/episode_02.speakers.txt \
  -o output/episode_02/
```

---

## Configuration

Edit `config.yaml` to change defaults:

```yaml
whisper:
  model: large-v3        # large-v3 or large-v3-turbo (faster, slightly less accurate)
  language: uk           # uk = Ukrainian, en = English

diarization:
  min_speakers: 4        # adjust to your show format
  max_speakers: 6

denoise:
  enabled: true          # set to false to skip Demucs globally
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `conda activate transcription` first |
| `HF_TOKEN not set` | Add your token to `.env` file |
| Wrong speaker names | Edit `speakers.txt` — reorder lines to match first-appearance order |
| Speaker assigned to wrong person | Tweak `--min-speakers` / `--max-speakers` |
| Out of VRAM | Set `compute_type: int8` in `config.yaml` |
| Slow processing | Use `model: large-v3-turbo` in `config.yaml` |

