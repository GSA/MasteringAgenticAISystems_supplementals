"""
Audio Processing Pipeline for Multimodal RAG
Converts meeting recordings to time-indexed searchable chunks
"""

import whisper
import torch
from pathlib import Path
from typing import List, Dict
import json

class WhisperAudioProcessor:
    """
    Processes audio recordings for RAG retrieval with timestamp preservation
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model

        Args:
            model_size: Options - "tiny" (39M), "base" (74M), "small" (244M),
                       "medium" (769M), "large" (1550M)
                       Larger models = higher accuracy, slower processing
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("Model loaded successfully")

    def transcribe(self, audio_path: str, language: str = "en") -> Dict:
        """
        Transcribe audio file to text with timestamps

        Args:
            audio_path: Path to audio file (supports mp3, wav, m4a, flac, ogg)
            language: ISO language code ("en", "es", "zh", "fr", etc.)
                     or None for auto-detection

        Returns:
            {
                "text": "complete transcript as single string",
                "segments": [
                    {
                        "start": 0.0,    # Start time in seconds
                        "end": 5.24,     # End time in seconds
                        "text": "Today we're discussing multimodal RAG."
                    },
                    ...
                ],
                "language": "en"  # Detected or specified language
            }
        """
        print(f"Transcribing {audio_path}...")
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",  # Alternative: "translate" to translate to English
            verbose=False,
            word_timestamps=False  # Sentence-level is sufficient for RAG
        )
        print(f"Transcription complete: {len(result['segments'])} segments")
        return result

    def create_time_indexed_chunks(
        self,
        audio_path: str,
        chunk_duration: int = 300,
        language: str = "en"
    ) -> List[Dict]:
        """
        Create time-indexed text chunks optimized for RAG retrieval

        Args:
            audio_path: Path to audio file
            chunk_duration: Target seconds per chunk (default 300 = 5 minutes)
                          Shorter = more precise retrieval, less context
                          Longer = more context, less precise retrieval
            language: Language code for transcription

        Returns:
            List of chunks with metadata:
            [
                {
                    "text": "chunk transcript text",
                    "start_time": 0.0,
                    "end_time": 305.2,
                    "audio_source": "path/to/audio.mp3",
                    "chunk_index": 0,
                    "metadata": {
                        "type": "audio_transcript",
                        "language": "en",
                        "duration": 305.2
                    }
                },
                ...
            ]
        """
        # Transcribe audio
        transcript = self.transcribe(audio_path, language=language)

        chunks = []
        current_chunk_text = []
        chunk_start_time = 0
        chunk_index = 0

        for segment in transcript["segments"]:
            current_chunk_text.append(segment["text"].strip())

            # Check if we've exceeded target duration
            segment_end = segment["end"]
            elapsed_time = segment_end - chunk_start_time

            if elapsed_time >= chunk_duration:
                # Finalize current chunk
                chunk = {
                    "text": " ".join(current_chunk_text),
                    "start_time": chunk_start_time,
                    "end_time": segment_end,
                    "audio_source": audio_path,
                    "chunk_index": chunk_index,
                    "metadata": {
                        "type": "audio_transcript",
                        "language": transcript["language"],
                        "duration": segment_end - chunk_start_time
                    }
                }
                chunks.append(chunk)

                # Reset for next chunk
                current_chunk_text = []
                chunk_start_time = segment_end
                chunk_index += 1

        # Capture remaining segments as final chunk
        if current_chunk_text:
            final_segment = transcript["segments"][-1]
            chunk = {
                "text": " ".join(current_chunk_text),
                "start_time": chunk_start_time,
                "end_time": final_segment["end"],
                "audio_source": audio_path,
                "chunk_index": chunk_index,
                "metadata": {
                    "type": "audio_transcript",
                    "language": transcript["language"],
                    "duration": final_segment["end"] - chunk_start_time
                }
            }
            chunks.append(chunk)

        print(f"Created {len(chunks)} chunks from {audio_path}")
        return chunks
