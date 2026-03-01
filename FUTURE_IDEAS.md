# Future Ideas

- **LLM post-processing** — pipe transcript through Ollama (Llama 3.1 8B / Gemma 2 9B) to fix punctuation, grammar, and formatting without changing meaning. Chunk by speaker blocks to preserve context.

- **Diarization accuracy tuning** — if speakers are getting mixed up, experiment with pyannote pipeline parameters: `min_duration_on`, `min_duration_off`, and clustering threshold. Also try `exclusive_speaker_diarization` output instead of `speaker_diarization`.

- **Batch processing** — add a `batch.py` script that reads a list of episode names and runs the pipeline sequentially, logging results and skipping already-processed episodes.

- **Web UI** — if CLI becomes inconvenient, a simple Gradio or Streamlit interface could let you drop a video file and download the transcript without touching the terminal.
