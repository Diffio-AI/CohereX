# CohereX

CohereX is a WhisperX-shaped transcription package built around `CohereLabs/cohere-transcribe-03-2026`.

The initial implementation keeps WhisperX-style output, pyannote VAD, and alignment, while replacing the Whisper ASR path with a local Cohere model implementation and VAD-first chunking.
