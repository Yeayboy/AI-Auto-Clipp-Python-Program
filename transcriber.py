"""
transcriber.py
==============
Modul integrasi Deepgram Nova-2 API untuk ekstraksi transkrip
dengan word-level timestamps dan speaker diarization.
"""

import os
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Word:
    word: str
    start: float          # detik
    end: float            # detik
    confidence: float
    speaker: Optional[int] = None   # 0, 1, 2, … (hasil diarization)

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "speaker": self.speaker,
        }


@dataclass
class TranscriptResult:
    full_text: str
    words: list[Word] = field(default_factory=list)
    speakers_detected: int = 1
    duration_seconds: float = 0.0
    language: str = "id"

    def to_dict(self) -> dict:
        return {
            "full_text": self.full_text,
            "words": [w.to_dict() for w in self.words],
            "speakers_detected": self.speakers_detected,
            "duration_seconds": self.duration_seconds,
            "language": self.language,
        }

    def get_segment_text(self, start: float, end: float) -> str:
        """Kembalikan teks dari rentang waktu tertentu."""
        segment_words = [w.word for w in self.words if start <= w.start <= end]
        return " ".join(segment_words)

    def get_segment_words(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if start <= w.start <= end]


# ---------------------------------------------------------------------------
# Audio Extractor (FFmpeg helper)
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, output_wav: Optional[str] = None) -> str:
    """
    Pisahkan audio dari file video menjadi 16kHz Mono WAV
    menggunakan FFmpeg agar upload ke API STT lebih ringan.

    Returns:
        Path ke file WAV yang dihasilkan.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")

    if output_wav is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_wav = tmp.name
        tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16 kHz
        "-ac", "1",               # mono
        output_wav,
    ]

    logger.info("Mengekstrak audio → %s", output_wav)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg gagal mengekstrak audio:\n{result.stderr}")

    logger.info("Ekstraksi audio selesai: %s", output_wav)
    return output_wav


# ---------------------------------------------------------------------------
# Deepgram Transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    """
    Wrapper Deepgram Nova-2 yang mengembalikan TranscriptResult
    dengan word-level timestamps dan speaker diarization.
    """

    MODEL = "nova-2"
    MAX_SPEAKERS = 2        # podcast/webinar biasanya maks 2 pembicara

    def __init__(self, api_key: Optional[str] = None):
        raw_api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self.api_key = raw_api_key.strip()
        if not self.api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY tidak ditemukan. "
                "Set environment variable atau pass api_key."
            )

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def transcribe_video(self, video_path: str) -> TranscriptResult:
        """
        Pipeline lengkap: extract audio → upload ke Deepgram → parse hasil.

        Args:
            video_path: Path ke file MP4/MOV.

        Returns:
            TranscriptResult berisi teks + word timestamps.
        """
        wav_path = None
        try:
            logger.info("Memulai transkripsi untuk: %s", video_path)
            wav_path = extract_audio(video_path)
            result = self._transcribe_audio_file(wav_path)
            return result
        finally:
            # Hapus file WAV sementara
            if wav_path and Path(wav_path).exists():
                Path(wav_path).unlink()
                logger.debug("File WAV sementara dihapus: %s", wav_path)

    def transcribe_audio(self, audio_path: str) -> TranscriptResult:
        """
        Transkripsi langsung dari file audio (WAV/MP3/M4A).
        """
        return self._transcribe_audio_file(audio_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transcribe_audio_file(self, audio_path: str) -> TranscriptResult:
        audio_path = Path(audio_path)
        logger.info("Mengirim %s ke Deepgram API …", audio_path.name)

        params = {
            "model": self.MODEL,
            "smart_format": "true",
            "utterances": "true",
            "punctuate": "true",
            "diarize": "true",
            "diarize_version": "2",
            "language": "id",
            "words": "true",
        }

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        for attempt in range(1, 4):
            try:
                timeout_config = httpx.Timeout(
                    connect=30.0,
                    read=300.0,
                    write=300.0,
                    pool=30.0,
                )
                with httpx.Client(timeout=timeout_config) as client:
                    response = client.post(
                        "https://api.deepgram.com/v1/listen",
                        params=params,
                        headers={
                            "Authorization": f"Token {self.api_key}",
                            "Content-Type": "audio/wav",
                        },
                        content=audio_bytes,
                    )

                response.raise_for_status()
                return self._parse_response(response.json())

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code not in (408, 429) and not 500 <= status_code < 600:
                    detail = exc.response.text.strip()[:500]
                    raise RuntimeError(
                        f"Deepgram API HTTP {status_code}: {detail or exc}"
                    ) from exc
                if attempt == 3:
                    detail = exc.response.text.strip()[:500]
                    raise RuntimeError(
                        f"Deepgram API HTTP {status_code} setelah 3 percobaan: "
                        f"{detail or exc}"
                    ) from exc
                logger.warning(
                    "Deepgram mengembalikan HTTP %d pada percobaan %d/3. "
                    "Mencoba ulang...",
                    status_code,
                    attempt,
                )
                time.sleep(5 * attempt)
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.TimeoutException) as exc:
                logger.warning(
                    "Deepgram timed out pada percobaan %d/3. Menunggu lalu retry... (%s)",
                    attempt,
                    exc,
                )
                if attempt == 3:
                    raise RuntimeError(f"Deepgram API error: {exc}") from exc
                time.sleep(5 * attempt)
            except Exception as exc:
                raise RuntimeError(f"Deepgram API error: {exc}") from exc

        raise RuntimeError("Deepgram API error: retry limit exceeded")

    def _parse_response(self, response) -> TranscriptResult:
        """Parse respons Deepgram ke TranscriptResult."""
        try:
            # Support both legacy SDK objects and REST JSON dictionaries.
            if hasattr(response, "results"):
                data = response
            else:
                data = response.get("results")
                if data is None:
                    raise RuntimeError("Response tidak memiliki field results.")
                data = response

            channel = data["results"]["channels"][0] if isinstance(data, dict) else data.results.channels[0]
            alternative = channel["alternatives"][0] if isinstance(channel, dict) else channel.alternatives[0]
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Struktur respons Deepgram tidak terduga: {exc}")

        full_text: str = alternative.get("transcript") if isinstance(alternative, dict) else (alternative.transcript or "")
        raw_words = alternative.get("words") if isinstance(alternative, dict) else (alternative.words or [])
        duration: float = (response.get("metadata", {}).get("duration") if isinstance(response, dict) else response.metadata.duration) if hasattr(response, "metadata") or isinstance(response, dict) else 0.0

        words: list[Word] = []
        speakers_seen: set[int] = set()

        for w in raw_words:
            if isinstance(w, dict):
                word_text = w.get("word", "")
                start = w.get("start", 0.0)
                end = w.get("end", 0.0)
                confidence = w.get("confidence", 0.0)
                speaker_id = w.get("speaker")
            else:
                word_text = w.word
                start = w.start
                end = w.end
                confidence = w.confidence
                speaker_id = getattr(w, "speaker", None)

            if speaker_id is not None:
                speakers_seen.add(int(speaker_id))

            words.append(Word(
                word=word_text,
                start=float(start),
                end=float(end),
                confidence=float(confidence),
                speaker=int(speaker_id) if speaker_id is not None else None,
            ))

        result = TranscriptResult(
            full_text=full_text,
            words=words,
            speakers_detected=len(speakers_seen) if speakers_seen else 1,
            duration_seconds=duration,
        )

        logger.info(
            "Transkripsi selesai: %d kata, %d pembicara, durasi %.1fs",
            len(words), result.speakers_detected, duration,
        )
        return result


# ---------------------------------------------------------------------------
# CLI / Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <video_or_audio_path>")
        sys.exit(1)

    transcriber = Transcriber()
    path = sys.argv[1]

    if path.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
        transcript = transcriber.transcribe_video(path)
    else:
        transcript = transcriber.transcribe_audio(path)

    output_file = Path(path).stem + "_transcript.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"✅ Transkrip disimpan ke: {output_file}")
    print(f"   Kata terdeteksi   : {len(transcript.words)}")
    print(f"   Pembicara         : {transcript.speakers_detected}")
    print(f"   Durasi            : {transcript.duration_seconds:.1f}s")
