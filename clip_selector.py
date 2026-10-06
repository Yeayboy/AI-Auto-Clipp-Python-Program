"""
clip_selector.py
================
LLM Decision Engine — menggunakan OpenAI GPT-4o-mini atau Claude 3.5 Haiku
untuk memilih momen viral berdasarkan transkrip + skor emosi Gemini,
dan mengembalikan JSON terstruktur sesuai kontrak PRD.
"""

import os
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from groq import Groq

from transcriber import TranscriptResult, Word
from multimodal_analyzer import MultimodalAnalysisResult, EmotionSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    GROQ   = "groq"
    CLAUDE = "claude"
    OPENAI = "openai"


GROQ_MODEL = "openai/gpt-oss-20b"
CLAUDE_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-4o-mini"

MIN_CLIP_DURATION = 25.0   # detik minimum
MAX_CLIP_DURATION = 65.0   # detik maksimum
MAX_CLIPS         = 3


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ClipInstruction:
    clip_id: str
    headline: str           # Hook Headline untuk overlay video
    start_time: float       # detik
    end_time:   float       # detik
    virality_score: float   # 0.0 – 1.0
    hook_statement: str     # Kalimat hook 3 detik pertama
    reason: str             # Justifikasi pemilihan

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "headline": self.headline,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": round(self.duration, 2),
            "virality_score": self.virality_score,
            "hook_statement": self.hook_statement,
            "reason": self.reason,
        }


@dataclass
class ClipSelectionResult:
    total_clips_found: int
    clips: list[ClipInstruction] = field(default_factory=list)
    provider_used: str = ""
    raw_llm_response: str = ""

    def to_dict(self) -> dict:
        return {
            "total_clips_found": self.total_clips_found,
            "clips": [c.to_dict() for c in self.clips],
            "provider_used": self.provider_used,
        }


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Anda adalah Video Editor profesional spesialis TikTok, Shorts, dan Reels.

Tugas Anda: Analisis transkrip teks berstempel waktu dan hasil emosi multimodal berikut.
Pilih 1 hingga 3 klip video berdurasi 30–60 detik dengan potensi viral tertinggi.

Aturan Pemilihan:
1. Pilih momen dengan pembuka yang relevan dan menarik; jangan memaksakan rasa penasaran.
2. Jangan memotong di tengah kata atau kalimat yang menggantung.
3. Pastikan start_time dan end_time PRESISI mengikuti batas kata pada transkrip.
4. Durasi setiap klip minimal 25 detik dan maksimal 65 detik.
5. Headline harus merangkum topik, tindakan, atau informasi konkret yang benar-benar didukung isi klip.
6. Gunakan bahasa Indonesia yang natural, spesifik, dan maksimal 8 kata.
7. Jangan mengarang detail atau memakai clickbait seperti "GILA", "HEBOH", "RAHASIA", atau "TERBONGKAR" kecuali benar-benar didukung klip.
8. Jika konteks terbatas, pilih headline netral berdasarkan transkrip; jangan menebak.
9. Kembalikan output HANYA dalam format JSON murni tanpa komentar tambahan.

Format JSON yang diwajibkan:
{
  "total_clips_found": <int>,
  "clips": [
    {
      "clip_id": "clip_001",
    "headline": "<RINGKASAN KONKRET KAPITAL MAX 8 KATA>",
      "start_time": <float detik>,
      "end_time": <float detik>,
      "virality_score": <0.0-1.0>,
      "hook_statement": "<kalimat hook 3 detik pertama>",
      "reason": "<alasan pemilihan klip ini>"
    }
  ]
}
"""


def _build_user_prompt(
    transcript: TranscriptResult,
    emotion_analysis: MultimodalAnalysisResult,
    max_words: int = 8000,
) -> str:
    """
    Susun prompt user berisi transkrip terformat + ringkasan emosi.
    Dipotong pada max_words untuk menghindari context window overflow.
    """
    # -- Transkrip dengan timestamps --
    word_lines: list[str] = []
    for w in transcript.words[:max_words]:
        spk = f"[S{w.speaker}]" if w.speaker is not None else ""
        word_lines.append(f"{w.start:.2f}s {spk} {w.word}")

    transcript_block = "\n".join(word_lines)
    truncated_note = ""
    if len(transcript.words) > max_words:
        truncated_note = f"\n[... +{len(transcript.words) - max_words} kata terpotong karena panjang ...]"

    # -- Ringkasan segmen emosi Gemini --
    emotion_lines: list[str] = []
    for seg in emotion_analysis.segments:
        tags = ", ".join(seg.emotion_tags) if seg.emotion_tags else "none"
        emotion_lines.append(
            f"{seg.start:.1f}s–{seg.end:.1f}s | energi={seg.energy_score:.2f} | "
            f"tags=[{tags}] | {seg.description}"
        )

    emotion_block = "\n".join(emotion_lines) if emotion_lines else "(tidak tersedia)"

    return f"""=== TRANSKRIP WORD-LEVEL (total: {len(transcript.words)} kata, {transcript.duration_seconds:.0f}s) ===
{transcript_block}{truncated_note}

=== ANALISIS EMOSI GEMINI (model: {emotion_analysis.model_used}) ===
Energi keseluruhan: {emotion_analysis.overall_energy:.2f}
{emotion_block}

=== INSTRUKSI ===
Pilih 1–3 klip terbaik. Perhatikan sinkronisasi kata pada transkrip.
Kembalikan JSON saja, tanpa penjelasan tambahan.
"""


# ---------------------------------------------------------------------------
# Clip Selector
# ---------------------------------------------------------------------------

class ClipSelector:
    """
    Kirim transkrip + data emosi ke LLM (Claude atau OpenAI)
    dan kembalikan daftar klip terstruktur.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 5.0

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.GROQ,
        groq_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.provider = provider

        if provider == LLMProvider.GROQ:
            api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY tidak ditemukan.")
            self._groq = Groq(api_key=api_key)

        elif provider == LLMProvider.CLAUDE:
            import anthropic

            api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY tidak ditemukan.")
            self._claude = anthropic.Anthropic(api_key=api_key)

        elif provider == LLMProvider.OPENAI:
            import openai

            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY tidak ditemukan.")
            self._openai = openai.OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def select_clips(
        self,
        transcript: TranscriptResult,
        emotion_analysis: MultimodalAnalysisResult,
    ) -> ClipSelectionResult:
        """
        Pilih klip viral terbaik menggunakan LLM.

        Args:
            transcript:       Hasil Deepgram (TranscriptResult).
            emotion_analysis: Hasil analisis Gemini (MultimodalAnalysisResult).

        Returns:
            ClipSelectionResult berisi daftar ClipInstruction siap render.
        """
        user_prompt = _build_user_prompt(transcript, emotion_analysis)

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(
                    "Mengirim ke LLM [%s] percobaan %d/%d …",
                    self.provider.value, attempt, self.MAX_RETRIES,
                )

                if self.provider == LLMProvider.GROQ:
                    raw = self._call_groq(user_prompt)
                elif self.provider == LLMProvider.CLAUDE:
                    raw = self._call_claude(user_prompt)
                else:
                    raw = self._call_openai(user_prompt)

                result = self._parse_llm_response(raw, transcript)
                result.provider_used = self.provider.value
                result.raw_llm_response = raw

                logger.info(
                    "LLM memilih %d klip (provider: %s)",
                    result.total_clips_found, result.provider_used,
                )
                return result

            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Parsing JSON LLM gagal (percobaan %d): %s", attempt, exc)
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        f"LLM tidak mengembalikan JSON valid setelah {self.MAX_RETRIES} percobaan."
                    ) from exc
                time.sleep(self.RETRY_DELAY)

            except Exception as exc:
                logger.error("LLM API error: %s", exc)
                if attempt == self.MAX_RETRIES:
                    raise
                time.sleep(self.RETRY_DELAY * attempt)

        return ClipSelectionResult(total_clips_found=0)

    # ------------------------------------------------------------------
    # LLM Callers
    # ------------------------------------------------------------------

    def _call_groq(self, user_prompt: str) -> str:
        response = self._groq.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=6000,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _call_claude(self, user_prompt: str) -> str:
        response = self._claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()

    def _call_openai(self, user_prompt: str) -> str:
        response = self._openai.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # JSON Parser & Validator
    # ------------------------------------------------------------------

    def _parse_llm_response(
        self, raw: str, transcript: TranscriptResult
    ) -> ClipSelectionResult:
        """Parse dan validasi JSON yang dikembalikan LLM."""
        # Bersihkan fence markdown jika ada
        clean = raw.strip()
        for fence in ("```json", "```"):
            if clean.startswith(fence):
                clean = clean[len(fence):]
            if clean.endswith("```"):
                clean = clean[:-3]
        clean = clean.strip()

        data = json.loads(clean)

        clips: list[ClipInstruction] = []
        for i, clip_data in enumerate(data.get("clips", [])[:MAX_CLIPS]):
            start  = float(clip_data["start_time"])
            end    = float(clip_data["end_time"])
            score  = float(clip_data.get("virality_score", 0.7))

            # Snap ke boundary kata terdekat dari transkrip
            start = self._snap_to_word_boundary(start, transcript, mode="start")
            end   = self._snap_to_word_boundary(end, transcript, mode="end")

            # Validasi durasi
            duration = end - start
            if duration < MIN_CLIP_DURATION:
                end = start + MIN_CLIP_DURATION
            elif duration > MAX_CLIP_DURATION:
                end = start + MAX_CLIP_DURATION

            # Clamp ke durasi video
            end = min(end, transcript.duration_seconds)
            if end <= start:
                logger.warning("Klip %s tidak valid (start=%.2f end=%.2f), dilewati.", clip_data.get("clip_id"), start, end)
                continue

            headline_words = str(clip_data.get("headline") or "").split()
            headline = " ".join(headline_words[:8]).upper()

            clips.append(ClipInstruction(
                clip_id=clip_data.get("clip_id", f"clip_{i+1:03d}"),
                headline=headline,
                start_time=round(start, 3),
                end_time=round(end, 3),
                virality_score=max(0.0, min(1.0, score)),
                hook_statement=clip_data.get("hook_statement", ""),
                reason=clip_data.get("reason", ""),
            ))

        # Urutkan berdasarkan virality_score tertinggi
        clips.sort(key=lambda c: c.virality_score, reverse=True)

        return ClipSelectionResult(
            total_clips_found=len(clips),
            clips=clips,
        )

    def _snap_to_word_boundary(
        self, time_sec: float, transcript: TranscriptResult, mode: str = "start"
    ) -> float:
        """
        Snap waktu ke batas kata terdekat di transkrip agar
        tidak memotong di tengah kata.
        """
        if not transcript.words:
            return time_sec

        tolerance = 2.0  # detik — cari kata dalam radius ini
        candidates: list[Word] = [
            w for w in transcript.words
            if abs((w.start if mode == "start" else w.end) - time_sec) <= tolerance
        ]

        if not candidates:
            return time_sec

        if mode == "start":
            closest = min(candidates, key=lambda w: abs(w.start - time_sec))
            return closest.start
        else:
            closest = min(candidates, key=lambda w: abs(w.end - time_sec))
            return closest.end


# ---------------------------------------------------------------------------
# CLI / Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python clip_selector.py <transcript.json> <emotions.json> [claude|openai]")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        t_data = json.load(f)
    with open(sys.argv[2], encoding="utf-8") as f:
        e_data = json.load(f)

    # Rekonstruksi objek dari dict
    from transcriber import Word as W
    from multimodal_analyzer import EmotionSegment as ES

    transcript = TranscriptResult(
        full_text=t_data["full_text"],
        words=[W(**w) for w in t_data["words"]],
        speakers_detected=t_data["speakers_detected"],
        duration_seconds=t_data["duration_seconds"],
    )
    emotions = MultimodalAnalysisResult(
        segments=[ES(**s) for s in e_data["segments"]],
        overall_energy=e_data["overall_energy"],
        model_used=e_data["model_used"],
        fallback_used=e_data["fallback_used"],
    )

    provider = LLMProvider(sys.argv[3]) if len(sys.argv) > 3 else LLMProvider.GROQ
    selector = ClipSelector(provider=provider)
    result   = selector.select_clips(transcript, emotions)

    output_file = "clips_selected.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"✅ {result.total_clips_found} klip dipilih → {output_file}")
    for c in result.clips:
        print(f"   [{c.clip_id}] {c.start_time:.1f}s–{c.end_time:.1f}s "
              f"({c.duration:.0f}s) score={c.virality_score:.2f} | {c.headline}")
