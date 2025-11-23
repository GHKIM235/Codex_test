"""Whisper transcription service that assumes Japanese audio input."""

from typing import Any, Dict, List

import whisper
from tqdm.auto import tqdm

from .audio_chunker import AudioChunk


class WhisperTranscriber:
    """Thin wrapper around OpenAI Whisper for chunked transcription."""

    def __init__(self, model_name: str = "small"):
        self.model_name = model_name
        self.model = whisper.load_model(model_name)

    def transcribe_chunks(self, chunks: List[AudioChunk]) -> List[Dict[str, Any]]:
        """
        Run transcription for each chunk and normalize timestamps back to the
        original audio timeline.
        """
        results: List[Dict[str, Any]] = []

        if not chunks:
            return results

        total_chunks = len(chunks)
        progress_bar = tqdm(
            chunks,
            total=total_chunks,
            desc="Transcribing chunks",
            unit="chunk",
        )

        for index, chunk in enumerate(progress_bar, start=1):
            progress_bar.set_postfix_str(f"chunk {index}/{total_chunks}")

            transcription = self.model.transcribe(
                str(chunk.path),
                language="ja",
                task="transcribe",
            )

            for segment in transcription.get("segments", []):
                start = chunk.start_time + float(segment["start"])
                end = chunk.start_time + float(segment["end"])
                text = segment["text"].strip()
                if not text:
                    continue
                results.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text,
                    }
                )

        progress_bar.close()
        return results
