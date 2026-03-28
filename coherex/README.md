# CohereX

CohereX is a WhisperX-shaped transcription package built around `CohereLabs/cohere-transcribe-03-2026`.

The initial implementation keeps WhisperX-style output, pyannote VAD, and alignment, while replacing the Whisper ASR path with a local Cohere model implementation and VAD-first chunking.

## Run With uv

From the repository root:

```bash
uv sync --project coherex --extra dev
uv run --project coherex coherex audio.mp3
```
