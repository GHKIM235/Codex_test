"""Command-line entry point for generating Korean subtitles from Japanese audio."""

import argparse
from pathlib import Path

from services.audio_extractor import extract_audio
from services.audio_chunker import chunk_audio
from services.transcriber import WhisperTranscriber
from services.translator import JapaneseToKoreanTranslator
from services.srt_writer import write_srt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Korean subtitles (SRT) from a Japanese mp4 video.",
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input mp4 video file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="Whisper model size to load (default: small).",
    )
    return parser


def run_pipeline(video_path: Path, model_name: str) -> Path:
    """
    Execute the end-to-end pipeline: extract audio, chunk, transcribe, translate, write SRT.
    """
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    working_dir = video_path.parent / f"{video_path.stem}_work"
    chunks_dir = working_dir / "chunks"

    audio_path = extract_audio(video_path, working_dir)
    chunks = chunk_audio(audio_path, chunks_dir)

    if not chunks:
        raise RuntimeError("No audio chunks were generated.")

    transcriber = WhisperTranscriber(model_name=model_name)
    segments = transcriber.transcribe_chunks(chunks)

    translator = JapaneseToKoreanTranslator()
    translated_segments = translator.translate_segments(segments)

    output_path = video_path.with_name(f"{video_path.stem}_ko.srt")
    write_srt(translated_segments, output_path)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    output_path = run_pipeline(Path(args.video_path), args.model)
    print(f"Korean subtitles saved to: {output_path}")


if __name__ == "__main__":
    main()

