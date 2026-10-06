"""
main.py
=======
Pipeline utama AI Auto-Clipper & Video Editor Engine.

Alur kerja:
  Input Video
    ↓
  [1] Transcriber  — Deepgram Nova-2 → word-level timestamps + diarization
    ↓
  [2] Multimodal Analyzer — Gemini 1.5 Flash → emotion/energy scoring
    ↓
  [3] Clip Selector — Groq / Claude / GPT-4o-mini → JSON klip viral
    ↓
  [4] Video Editor  — FFmpeg → crop 9:16 + subtitle ASS + render MP4
    ↓
  Output Klip MP4 Final

Penggunaan:
  python main.py --input video.mp4 --output output/ [--llm groq|claude|openai]
  python main.py --input video.mp4 --skip-gemini   (fallback analisis teks)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from transcriber import Transcriber, TranscriptResult
from multimodal_analyzer import MultimodalAnalyzer, MultimodalAnalysisResult
from clip_selector import ClipSelector, ClipSelectionResult, LLMProvider
from video_editor import VideoEditor, RenderConfig, RenderResult

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Config & Result
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    input_video: str
    output_dir: str = "output"
    llm_provider: LLMProvider = LLMProvider.GROQ
    skip_gemini: bool = False       # Paksa fallback teks untuk analisis emosi
    save_intermediates: bool = True  # Simpan hasil JSON antar-tahap
    log_level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class PipelineResult:
    input_video: str
    total_clips: int
    successful_clips: int
    failed_clips: int
    output_files: list[str] = field(default_factory=list)
    stage_times: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.successful_clips > 0

    def print_summary(self):
        print("\n" + "="*60)
        print("📊 RINGKASAN PIPELINE AI AUTO-CLIPPER")
        print("="*60)
        print(f"  Input       : {self.input_video}")
        print(f"  Klip total  : {self.total_clips}")
        print(f"  Berhasil    : {self.successful_clips}")
        print(f"  Gagal       : {self.failed_clips}")
        print()

        if self.stage_times:
            print("⏱  Waktu per Tahap:")
            for stage, t in self.stage_times.items():
                print(f"    {stage:<25} {t:.1f}s")
            print(f"    {'TOTAL':<25} {sum(self.stage_times.values()):.1f}s")
            print()

        if self.output_files:
            print("📁 File Output:")
            for f in self.output_files:
                size_mb = Path(f).stat().st_size / (1024 * 1024) if Path(f).exists() else 0
                print(f"    {f}  ({size_mb:.1f} MB)")

        if self.error:
            print(f"\n❌ Error: {self.error}")
        print("="*60)


# ---------------------------------------------------------------------------
# Pipeline Class
# ---------------------------------------------------------------------------

class AIClipperPipeline:
    """
    Orkestrasi penuh 4-tahap pipeline AI Auto-Clipper.
    Mendukung cache JSON antar-tahap untuk mempersingkat waktu debugging.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_root = self.output_dir / ".cache"
        self.cache_root.mkdir(exist_ok=True)
        self.cache_dir = self.cache_root
        self._stage_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public Runner
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """Jalankan pipeline penuh dari awal hingga akhir."""
        video_path = Path(self.config.input_video)
        if not video_path.exists():
            return PipelineResult(
                input_video=str(video_path),
                total_clips=0,
                successful_clips=0,
                failed_clips=0,
                error=f"File tidak ditemukan: {video_path}",
            )

        self.cache_dir = self.cache_root / self._cache_key(video_path)
        self.cache_dir.mkdir(exist_ok=True)

        logger.info("="*55)
        logger.info("🎬 AI Auto-Clipper Pipeline dimulai")
        logger.info("   Input  : %s", video_path)
        logger.info("   Output : %s", self.output_dir)
        logger.info("="*55)

        try:
            # --- Tahap 1: Transkripsi ---
            transcript = self._stage_transcribe(str(video_path))

            # --- Tahap 2: Analisis Multimodal ---
            emotions = self._stage_analyze(str(video_path), transcript)

            # --- Tahap 3: Seleksi Klip ---
            selection = self._stage_select(transcript, emotions)

            if selection.total_clips_found == 0:
                logger.warning("LLM tidak menemukan klip yang memenuhi syarat.")
                return PipelineResult(
                    input_video=str(video_path),
                    total_clips=0,
                    successful_clips=0,
                    failed_clips=0,
                    stage_times=self._stage_times,
                    error="Tidak ada klip yang dipilih oleh LLM.",
                )

            # --- Tahap 4: Render Video ---
            render_results = self._stage_render(
                str(video_path), transcript, selection
            )

            # Rangkum hasil
            successful = [r for r in render_results if r.success]
            failed     = [r for r in render_results if not r.success]

            return PipelineResult(
                input_video=str(video_path),
                total_clips=len(render_results),
                successful_clips=len(successful),
                failed_clips=len(failed),
                output_files=[r.output_path for r in successful],
                stage_times=self._stage_times,
            )

        except Exception as exc:
            logger.exception("Pipeline berhenti karena error: %s", exc)
            return PipelineResult(
                input_video=str(video_path),
                total_clips=0,
                successful_clips=0,
                failed_clips=0,
                stage_times=self._stage_times,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Stage 1: Transcription
    # ------------------------------------------------------------------

    def _stage_transcribe(self, video_path: str) -> TranscriptResult:
        cache_path = self.cache_dir / "transcript.json"

        # Coba load dari cache
        if cache_path.exists() and self.config.save_intermediates:
            logger.info("[1/4] 📝 Memuat transkrip dari cache …")
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            return self._dict_to_transcript(data)

        logger.info("[1/4] 📝 Memulai transkripsi Deepgram …")
        t0 = time.time()

        transcriber = Transcriber()
        transcript  = transcriber.transcribe_video(video_path)

        elapsed = time.time() - t0
        self._stage_times["1_transcription"] = elapsed
        logger.info("[1/4] ✅ Transkripsi selesai dalam %.1fs", elapsed)
        logger.info("      %d kata | %d pembicara | %.1fs durasi",
                    len(transcript.words), transcript.speakers_detected, transcript.duration_seconds)

        if self.config.save_intermediates:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)

        return transcript

    # ------------------------------------------------------------------
    # Stage 2: Multimodal Analysis
    # ------------------------------------------------------------------

    def _stage_analyze(
        self, video_path: str, transcript: TranscriptResult
    ) -> MultimodalAnalysisResult:
        cache_path = self.cache_dir / "emotions.json"

        if cache_path.exists() and self.config.save_intermediates:
            logger.info("[2/4] 🎭 Memuat hasil emosi dari cache …")
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            return self._dict_to_emotions(data)

        if self.config.skip_gemini:
            logger.info("[2/4] 🎭 Gemini dilewati — menggunakan fallback teks …")
            t0 = time.time()
            analyzer = MultimodalAnalyzer.__new__(MultimodalAnalyzer)
            result = analyzer._text_only_fallback(transcript, transcript.duration_seconds)
            self._stage_times["2_multimodal"] = time.time() - t0
        else:
            logger.info("[2/4] 🎭 Memulai analisis Gemini multimodal …")
            t0 = time.time()
            analyzer = MultimodalAnalyzer()
            result   = analyzer.analyze_video(
                video_path=video_path,
                duration_seconds=transcript.duration_seconds,
                fallback_transcript=transcript,
            )
            elapsed = time.time() - t0
            self._stage_times["2_multimodal"] = elapsed
            logger.info("[2/4] ✅ Analisis Gemini selesai dalam %.1fs (model: %s, fallback: %s)",
                        elapsed, result.model_used, result.fallback_used)

        if self.config.save_intermediates:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        return result

    # ------------------------------------------------------------------
    # Stage 3: Clip Selection
    # ------------------------------------------------------------------

    def _stage_select(
        self,
        transcript: TranscriptResult,
        emotions: MultimodalAnalysisResult,
    ) -> ClipSelectionResult:
        cache_path = self.cache_dir / "clips_selected.json"

        if cache_path.exists() and self.config.save_intermediates:
            logger.info("[3/4] 🤖 Memuat seleksi klip dari cache …")
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            return self._dict_to_selection(data)

        logger.info("[3/4] 🤖 Mengirim ke LLM Decision Engine (%s) …", self.config.llm_provider.value)
        t0 = time.time()

        selector  = ClipSelector(provider=self.config.llm_provider)
        selection = selector.select_clips(transcript, emotions)

        elapsed = time.time() - t0
        self._stage_times["3_llm_selection"] = elapsed
        logger.info("[3/4] ✅ LLM memilih %d klip dalam %.1fs",
                    selection.total_clips_found, elapsed)

        for clip in selection.clips:
            logger.info("      [%s] %.1fs–%.1fs (%.0fs) score=%.2f | %s",
                        clip.clip_id, clip.start_time, clip.end_time,
                        clip.duration, clip.virality_score, clip.headline)

        if self.config.save_intermediates:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(selection.to_dict(), f, ensure_ascii=False, indent=2)

        return selection

    # ------------------------------------------------------------------
    # Stage 4: Video Rendering
    # ------------------------------------------------------------------

    def _stage_render(
        self,
        video_path: str,
        transcript: TranscriptResult,
        selection: ClipSelectionResult,
    ) -> list[RenderResult]:
        logger.info("[4/4] 🎞  Memulai rendering %d klip …", selection.total_clips_found)
        t0 = time.time()

        editor  = VideoEditor(output_dir=str(self.output_dir))
        configs: list[RenderConfig] = []

        for clip in selection.clips:
            subtitle_words = transcript.get_segment_words(clip.start_time, clip.end_time)
            configs.append(RenderConfig(
                input_video=video_path,
                output_path=str(self.output_dir / f"{clip.clip_id}.mp4"),
                start_time=clip.start_time,
                end_time=clip.end_time,
                headline=clip.headline,
                subtitle_words=subtitle_words,
                speakers_detected=transcript.speakers_detected,
                clip_id=clip.clip_id,
            ))

        results = editor.render_all(configs)
        editor.close()

        elapsed = time.time() - t0
        self._stage_times["4_rendering"] = elapsed

        success_count = sum(1 for r in results if r.success)
        logger.info("[4/4] ✅ Rendering selesai: %d/%d klip berhasil (%.1fs)",
                    success_count, len(results), elapsed)

        for r in results:
            if r.success:
                logger.info("      ✅ %s → %s", r.clip_id, r.output_path)
            else:
                logger.error("      ❌ %s GAGAL: %s", r.clip_id, r.error_message)

        return results

    # ------------------------------------------------------------------
    # JSON ↔ Dataclass Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(video_path: Path) -> str:
        stat = video_path.stat()
        identity = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _dict_to_transcript(data: dict) -> TranscriptResult:
        from transcriber import Word, TranscriptResult
        return TranscriptResult(
            full_text=data["full_text"],
            words=[Word(**w) for w in data["words"]],
            speakers_detected=data["speakers_detected"],
            duration_seconds=data["duration_seconds"],
        )

    @staticmethod
    def _dict_to_emotions(data: dict) -> MultimodalAnalysisResult:
        from multimodal_analyzer import EmotionSegment, MultimodalAnalysisResult
        return MultimodalAnalysisResult(
            segments=[EmotionSegment(**s) for s in data["segments"]],
            overall_energy=data["overall_energy"],
            model_used=data["model_used"],
            fallback_used=data["fallback_used"],
        )

    @staticmethod
    def _dict_to_selection(data: dict) -> ClipSelectionResult:
        from clip_selector import ClipInstruction, ClipSelectionResult
        clips = [
            ClipInstruction(**{key: value for key, value in clip.items() if key != "duration"})
            for clip in data["clips"]
        ]
        return ClipSelectionResult(
            total_clips_found=data["total_clips_found"],
            clips=clips,
            provider_used=data.get("provider_used", ""),
        )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Auto-Clipper — Ekstrak klip viral dari video panjang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python main.py --input podcast.mp4
  python main.py --input podcast.mp4 --output hasil/ --llm openai
  python main.py --input podcast.mp4 --skip-gemini
  python main.py --input podcast.mp4 --no-cache
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Path ke file video input (MP4/MOV)")
    parser.add_argument("--output", "-o", default="output", help="Direktori output (default: output/)")
    parser.add_argument(
        "--llm",
        choices=["groq", "claude", "openai"],
        default="groq",
        help="LLM provider untuk Decision Engine (default: groq)",
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Lewati analisis Gemini, gunakan fallback teks saja",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Jangan simpan/baca cache JSON antar-tahap",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Level logging (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Simpan log ke file (opsional)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Hapus cache dari run sebelumnya sebelum mulai",
    )

    return parser.parse_args()


def main():
    load_dotenv()  # Baca .env file jika ada
    args = parse_args()
    setup_logging(args.log_level, args.log_file)

    # Validasi API keys yang dibutuhkan
    missing_keys: list[str] = []
    if not os.environ.get("DEEPGRAM_API_KEY", "").strip():
        missing_keys.append("DEEPGRAM_API_KEY")
    if not args.skip_gemini and not os.environ.get("GOOGLE_API_KEY", "").strip():
        missing_keys.append("GOOGLE_API_KEY")
    if args.llm == "groq" and not os.environ.get("GROQ_API_KEY", "").strip():
        missing_keys.append("GROQ_API_KEY")
    if args.llm == "claude" and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        missing_keys.append("ANTHROPIC_API_KEY")
    if args.llm == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
        missing_keys.append("OPENAI_API_KEY")

    if missing_keys:
        logger.error("❌ API key berikut belum diset: %s", ", ".join(missing_keys))
        logger.error("   Set di file .env atau sebagai environment variable.")
        sys.exit(1)

    # Hapus cache jika diminta
    if args.clear_cache:
        cache_dir = Path(args.output) / ".cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            logger.info("Cache dihapus: %s", cache_dir)

    config = PipelineConfig(
        input_video=args.input,
        output_dir=args.output,
        llm_provider=LLMProvider(args.llm),
        skip_gemini=args.skip_gemini,
        save_intermediates=not args.no_cache,
        log_level=args.log_level,
        log_file=args.log_file,
    )

    pipeline = AIClipperPipeline(config)
    result   = pipeline.run()
    result.print_summary()

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
