"""
multimodal_analyzer.py
=======================
Integrasi Google Gemini 1.5 Flash API untuk analisis multimodal
(visual + audio) guna mendeteksi momen energi tinggi pada video.
"""

import os
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None
    import google.generativeai as genai

from google.api_core.exceptions import ResourceExhausted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class EmotionSegment:
    """Satu segmen hasil analisis emosi Gemini."""
    start: float                      # detik
    end: float                        # detik
    energy_score: float               # 0.0 – 1.0
    emotion_tags: list[str] = field(default_factory=list)   # ["laughter", "surprise", …]
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "energy_score": self.energy_score,
            "emotion_tags": self.emotion_tags,
            "description": self.description,
        }


@dataclass
class MultimodalAnalysisResult:
    segments: list[EmotionSegment] = field(default_factory=list)
    overall_energy: float = 0.0
    model_used: str = ""
    fallback_used: bool = False       # True jika Gemini gagal & fallback ke STT

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "overall_energy": self.overall_energy,
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
        }

    def get_high_energy_segments(self, threshold: float = 0.6) -> list[EmotionSegment]:
        return [s for s in self.segments if s.energy_score >= threshold]


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """
Anda adalah analis konten video berpengalaman. Analisis video/audio berikut dan
identifikasi semua momen dengan energi tinggi yang berpotensi viral.

Kembalikan hasil HANYA sebagai JSON valid dengan struktur berikut (tanpa teks tambahan):
{
  "segments": [
    {
      "start": <detik float>,
      "end": <detik float>,
      "energy_score": <0.0-1.0 float>,
      "emotion_tags": ["<tag>"],
      "description": "<deskripsi singkat Bahasa Indonesia>"
    }
  ],
  "overall_energy": <0.0-1.0 float>
}

Tag emosi yang dikenali:
- laughter, excitement, surprise, anger, sadness,
  high_pitch, applause, crowd_reaction, dramatic_pause,
  key_statement, call_to_action

Pertimbangkan:
1. Lonjakan nada suara dan volume
2. Tawa atau reaksi keramaian
3. Ekspresi wajah dramatis
4. Gerakan gestur aktif
5. Pernyataan kunci atau pertanyaan retoris
6. Momen kejutan atau kontroversi
"""


# ---------------------------------------------------------------------------
# Gemini Multimodal Analyzer
# ---------------------------------------------------------------------------

class MultimodalAnalyzer:
    """
    Menganalisis video menggunakan Gemini 1.5 Flash untuk
    mendeteksi segmen berenergi tinggi secara multimodal.
    """

    MODEL_PRIMARY = "gemini-1.5-flash"
    MODEL_FALLBACK = "gemini-1.5-pro"
    CHUNK_DURATION = 300   # detik — analisis per chunk 5 menit
    MAX_RETRIES = 3
    RETRY_DELAY = 60       # detik saat rate-limited

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY tidak ditemukan. "
                "Set environment variable atau pass api_key."
            )

        self.client = None
        self._model_name = self.MODEL_PRIMARY

        if genai is not None and hasattr(genai, "Client"):
            self.client = genai.Client(api_key=self.api_key)
            self._model = self._model_name
        else:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.MODEL_PRIMARY)

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def analyze_video(
        self,
        video_path: str,
        duration_seconds: float,
        fallback_transcript=None,   # TranscriptResult — digunakan jika Gemini gagal
    ) -> MultimodalAnalysisResult:
        """
        Analisis penuh video untuk mendeteksi momen viral.

        Args:
            video_path:         Path ke file MP4/MOV.
            duration_seconds:   Total durasi video.
            fallback_transcript: TranscriptResult untuk fallback jika Gemini error.

        Returns:
            MultimodalAnalysisResult dengan daftar EmotionSegment.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")

        logger.info("Memulai analisis multimodal Gemini untuk: %s", video_path.name)

        # Upload video ke Gemini File API
        try:
            gemini_file = self._upload_video(video_path)
            result = self._analyze_full_video(gemini_file, duration_seconds)
            result.model_used = self.MODEL_PRIMARY
            return result

        except ResourceExhausted as exc:
            logger.warning("Gemini rate-limited. Mencoba model fallback: %s", exc)
            return self._try_fallback_model(video_path, duration_seconds, fallback_transcript)

        except Exception as exc:
            logger.error("Gemini analysis gagal: %s", exc)
            if fallback_transcript:
                logger.info("Menggunakan fallback analisis berbasis transkrip teks.")
                return self._text_only_fallback(fallback_transcript, duration_seconds)
            raise

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def _upload_video(self, video_path: Path):
        """Upload video ke Gemini File API dan tunggu sampai siap diproses."""
        logger.info("Mengunggah video ke Gemini File API …")

        if self.client is not None:
            gemini_file = self.client.files.upload(
                file=str(video_path),
                config={"mime_type": "video/mp4"},
            )

            while True:
                state = getattr(gemini_file, "state", None)
                state_name = getattr(state, "name", None) or str(state).upper()
                if state_name in {"ACTIVE", "SUCCEEDED", "READY"}:
                    break
                if state_name == "FAILED":
                    raise RuntimeError(f"Gemini gagal memproses file: {gemini_file.name}")
                logger.debug("File sedang diproses Gemini, tunggu 10 detik …")
                time.sleep(10)
                gemini_file = self.client.files.get(name=gemini_file.name)

            logger.info("Video berhasil diupload: %s", gemini_file.name)
            return gemini_file

        gemini_file = genai.upload_file(str(video_path), mime_type="video/mp4")

        while gemini_file.state.name == "PROCESSING":
            logger.debug("File sedang diproses Gemini, tunggu 10 detik …")
            time.sleep(10)
            gemini_file = genai.get_file(gemini_file.name)

        if gemini_file.state.name == "FAILED":
            raise RuntimeError(f"Gemini gagal memproses file: {gemini_file.name}")

        logger.info("Video berhasil diupload: %s", gemini_file.name)
        return gemini_file

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze_full_video(self, gemini_file, duration_seconds: float) -> MultimodalAnalysisResult:
        """Kirim request analisis ke Gemini dan parse hasilnya."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info("Analisis Gemini (percobaan %d/%d) …", attempt, self.MAX_RETRIES)

                if self.client is not None:
                    response = self.client.models.generate_content(
                        model=self._model_name,
                        contents=[gemini_file, ANALYSIS_PROMPT],
                        config={
                            "response_mime_type": "application/json",
                            "temperature": 0.2,
                        },
                    )
                else:
                    response = self._model.generate_content(
                        [gemini_file, ANALYSIS_PROMPT],
                        generation_config={
                            "response_mime_type": "application/json",
                            "temperature": 0.2,
                        },
                    )

                return self._parse_gemini_response(response.text, duration_seconds)

            except ResourceExhausted:
                if attempt < self.MAX_RETRIES:
                    logger.warning("Rate limited, menunggu %ds …", self.RETRY_DELAY)
                    time.sleep(self.RETRY_DELAY)
                else:
                    raise

            except json.JSONDecodeError as exc:
                logger.warning("JSON tidak valid dari Gemini (percobaan %d): %s", attempt, exc)
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError("Gemini tidak mengembalikan JSON valid.") from exc

        return MultimodalAnalysisResult()  # tidak akan tercapai

    def _try_fallback_model(
        self, video_path: Path, duration_seconds: float, fallback_transcript
    ) -> MultimodalAnalysisResult:
        """Coba model Gemini Pro jika Flash rate-limited."""
        try:
            if self.client is not None:
                self._model_name = self.MODEL_FALLBACK
            else:
                self._model = genai.GenerativeModel(self.MODEL_FALLBACK)
            gemini_file = self._upload_video(video_path)
            result = self._analyze_full_video(gemini_file, duration_seconds)
            result.model_used = self.MODEL_FALLBACK
            return result
        except Exception as exc:
            logger.error("Fallback Gemini Pro juga gagal: %s", exc)
            if fallback_transcript:
                return self._text_only_fallback(fallback_transcript, duration_seconds)
            raise

    # ------------------------------------------------------------------
    # Fallback: Analisis Berbasis Teks Saja
    # ------------------------------------------------------------------

    def _text_only_fallback(self, transcript, duration_seconds: float) -> MultimodalAnalysisResult:
        """
        Fallback sederhana menggunakan heuristik kata kunci dari transkrip
        ketika semua API Gemini tidak tersedia.
        """
        logger.info("Menjalankan fallback analisis teks …")

        HIGH_ENERGY_KEYWORDS = {
            "laughter":    ["haha", "hahaha", "hehe", "lucu", "ketawa"],
            "surprise":    ["wow", "gila", "gak nyangka", "serius", "beneran", "benaran"],
            "key_statement": ["penting", "rahasia", "ternyata", "faktanya", "terbukti"],
            "excitement":  ["semangat", "keren", "mantap", "luar biasa", "dahsyat"],
        }

        segment_duration = 30.0
        num_segments = max(1, math.ceil(duration_seconds / segment_duration))
        segments: list[EmotionSegment] = []

        for i in range(num_segments):
            seg_start = i * segment_duration
            seg_end = min((i + 1) * segment_duration, duration_seconds)
            seg_text = transcript.get_segment_text(seg_start, seg_end).lower()

            tags_found: list[str] = []
            score = 0.3  # base score

            for tag, keywords in HIGH_ENERGY_KEYWORDS.items():
                if any(kw in seg_text for kw in keywords):
                    tags_found.append(tag)
                    score = min(1.0, score + 0.15)

            segments.append(EmotionSegment(
                start=seg_start,
                end=seg_end,
                energy_score=round(score, 2),
                emotion_tags=tags_found,
                description="[Fallback] Analisis berbasis teks saja.",
            ))

        overall = sum(s.energy_score for s in segments) / len(segments) if segments else 0.0

        return MultimodalAnalysisResult(
            segments=segments,
            overall_energy=round(overall, 2),
            model_used="text_fallback",
            fallback_used=True,
        )

    # ------------------------------------------------------------------
    # Response Parser
    # ------------------------------------------------------------------

    def _parse_gemini_response(self, response_text: str, duration_seconds: float) -> MultimodalAnalysisResult:
        """Parse teks JSON dari Gemini ke MultimodalAnalysisResult."""
        # Bersihkan fence markdown jika ada
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])

        data = json.loads(clean.strip())

        segments: list[EmotionSegment] = []
        for seg in data.get("segments", []):
            # Clamp waktu agar tidak melebihi durasi video
            start = max(0.0, float(seg.get("start", 0)))
            end = min(duration_seconds, float(seg.get("end", start + 30)))

            segments.append(EmotionSegment(
                start=round(start, 3),
                end=round(end, 3),
                energy_score=max(0.0, min(1.0, float(seg.get("energy_score", 0.5)))),
                emotion_tags=seg.get("emotion_tags", []),
                description=seg.get("description", ""),
            ))

        overall = max(0.0, min(1.0, float(data.get("overall_energy", 0.5))))

        logger.info(
            "Analisis Gemini selesai: %d segmen ditemukan, energi keseluruhan %.2f",
            len(segments), overall,
        )

        return MultimodalAnalysisResult(
            segments=segments,
            overall_energy=overall,
        )


# ---------------------------------------------------------------------------
# CLI / Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python multimodal_analyzer.py <video_path> <duration_seconds>")
        sys.exit(1)

    analyzer = MultimodalAnalyzer()
    result = analyzer.analyze_video(sys.argv[1], float(sys.argv[2]))

    output_file = Path(sys.argv[1]).stem + "_emotions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"✅ Hasil analisis disimpan ke: {output_file}")
    print(f"   Segmen ditemukan   : {len(result.segments)}")
    print(f"   Segmen energi ≥0.6 : {len(result.get_high_energy_segments())}")
    print(f"   Model              : {result.model_used}")
    print(f"   Fallback           : {result.fallback_used}")
