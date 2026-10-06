"""
video_editor.py
===============
Engine pemrosesan video:
  1. Auto-crop 16:9 → 9:16 dengan face detection (center-crop / split-screen)
  2. Generasi file subtitle animasi .ASS gaya karaoke (word-by-word highlight)
  3. Trimming & rendering akhir ke MP4 via FFmpeg
"""

import os
import re
import json
import logging
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

from transcriber import Word

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

OUTPUT_WIDTH  = 1080
OUTPUT_HEIGHT = 1920

# Warna subtitle karaoke (.ASS BGR hex format)
COLOR_PRIMARY   = "&H00FFFFFF"   # Putih — kata belum diucapkan
COLOR_HIGHLIGHT = "&H0000E5FF"   # Kuning — kata sedang diucapkan
COLOR_OUTLINE   = "&H00000000"   # Hitam — outline teks

FONT_NAME = "Arial"
FONT_SIZE = 30
HEADLINE_FONT_SIZE = 40

VIDEO_CRF   = 23       # Kualitas encoding (lebih kecil = lebih baik, lebih besar file)
VIDEO_PRESET = "fast"  # ultrafast/fast/medium/slow


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class FacePosition:
    x: int        # koordinat tengah wajah
    y: int
    width: int
    height: int
    confidence: float = 1.0


@dataclass
class RenderConfig:
    input_video: str
    output_path: str
    start_time: float
    end_time: float
    headline: str
    subtitle_words: list[Word]
    speakers_detected: int = 1
    clip_id: str = "clip_001"


@dataclass
class RenderResult:
    clip_id: str
    output_path: str
    duration: float
    success: bool
    error_message: str = ""


# ---------------------------------------------------------------------------
# Face Detector
# ---------------------------------------------------------------------------

class FaceDetector:
    """Wrapper MediaPipe Face Detection untuk posisi wajah per frame."""

    def __init__(self, min_detection_confidence: float = 0.5):
        solutions = getattr(mp, "solutions", None)
        self._mp_face = getattr(solutions, "face_detection", None)
        self._detector = None
        if self._mp_face is None:
            logger.warning(
                "MediaPipe tidak menyediakan API solutions; "
                "pelacakan wajah dinonaktifkan dan center-crop akan digunakan."
            )
            return

        self._detector = self._mp_face.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, frame: np.ndarray) -> list[FacePosition]:
        """Deteksi wajah dalam frame BGR OpenCV."""
        if self._detector is None:
            return []

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        faces: list[FacePosition] = []

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                fx = int(bbox.xmin * w)
                fy = int(bbox.ymin * h)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                cx = fx + fw // 2
                cy = fy + fh // 2
                faces.append(FacePosition(
                    x=cx, y=cy,
                    width=fw, height=fh,
                    confidence=det.score[0] if det.score else 1.0,
                ))

        return faces

    def get_dominant_face_x(self, video_path: str, start_sec: float, end_sec: float) -> Optional[int]:
        """
        Sampel beberapa frame dari segmen untuk menentukan
        posisi X wajah dominan (untuk center-crop tracking).
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int((end_sec - start_sec) * fps)
        sample_count = min(10, total_frames)

        face_x_positions: list[int] = []

        for i in range(sample_count):
            frame_no = int(start_sec * fps) + i * max(1, total_frames // sample_count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                continue
            faces = self.detect_faces(frame)
            if faces:
                best = max(faces, key=lambda f: f.confidence)
                face_x_positions.append(best.x)

        cap.release()

        if not face_x_positions:
            return None

        median_x = int(np.median(face_x_positions))
        logger.debug("Wajah dominan terdeteksi di x=%d", median_x)
        return median_x

    def close(self):
        if self._detector is not None:
            self._detector.close()


# ---------------------------------------------------------------------------
# ASS Subtitle Generator
# ---------------------------------------------------------------------------

class ASSSubtitleGenerator:
    """
    Membuat file subtitle .ASS dengan animasi karaoke word-by-word.
    Setiap kata disorot dengan warna berbeda saat diucapkan.
    """

    HEADER_TEMPLATE = """\
[Script Info]
Title: AI Auto-Clipper Karaoke
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
Timer: 100.0000
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{font_name},{font_size},{color_primary},{color_highlight},{color_outline},&H80000000,-1,0,0,0,100,100,0,0,3,8,0,2,40,40,150,1
Style: Headline,{font_name},{headline_size},{color_highlight},{color_primary},{color_outline},&H80000000,-1,0,0,0,100,100,0,0,1,3,1,8,32,32,75,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def __init__(
        self,
        width: int = OUTPUT_WIDTH,
        height: int = OUTPUT_HEIGHT,
        font_name: str = FONT_NAME,
        font_size: int = FONT_SIZE,
        headline_font_size: int = HEADLINE_FONT_SIZE,
    ):
        self.width = width
        self.height = height
        self.font_name = font_name
        self.font_size = font_size
        self.headline_font_size = headline_font_size

    def generate(
        self,
        words: list[Word],
        headline: str,
        clip_start: float,
        clip_end: float,
        output_path: str,
    ) -> str:
        """
        Buat file .ASS subtitle.

        Args:
            words:       Daftar Word dengan timestamps absolut.
            headline:    Teks judul hook overlay.
            clip_start:  Waktu mulai klip (offset untuk normalisasi).
            clip_end:    Waktu akhir klip.
            output_path: Path output file .ASS.

        Returns:
            Path ke file .ASS yang dihasilkan.
        """
        header = self.HEADER_TEMPLATE.format(
            width=self.width,
            height=self.height,
            font_name=self.font_name,
            font_size=self.font_size,
            headline_size=self.headline_font_size,
            color_primary=COLOR_PRIMARY,
            color_highlight=COLOR_HIGHLIGHT,
            color_outline=COLOR_OUTLINE,
        )

        events: list[str] = []

        # -- Headline overlay (1/3 bagian atas, sepanjang klip) --
        events.append(self._make_headline_event(headline, 0.0, clip_end - clip_start))

        # -- Subtitle karaoke per baris (~8 kata per baris) --
        segment_words = [w for w in words if clip_start <= w.start <= clip_end]
        lines = self._group_words_into_lines(segment_words, max_words_per_line=6)

        for line_words in lines:
            if not line_words:
                continue
            line_start = line_words[0].start - clip_start
            line_end   = line_words[-1].end  - clip_start
            events.append(self._make_karaoke_line(line_words, line_start, line_end, clip_start))

        content = header + "\n".join(events)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        logger.debug("File subtitle dibuat: %s", output_path)
        return str(output_path)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ts(seconds: float) -> str:
        """Float detik → format ASS timestamp H:MM:SS.cc"""
        seconds = max(0.0, seconds)
        h  = int(seconds // 3600)
        m  = int((seconds % 3600) // 60)
        s  = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100))
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def _make_headline_event(self, headline: str, start: float, end: float) -> str:
        escaped = headline.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        escaped = r"\N".join(textwrap.wrap(escaped, width=32) or [escaped])
        return (
            f"Dialogue: 0,{self._ts(start)},{self._ts(end)},Headline,,0,0,0,,"
            f"{{\\an8}}{escaped}\n"
        )

    def _make_karaoke_line(
        self,
        words: list[Word],
        line_start: float,
        line_end: float,
        clip_start: float,
    ) -> str:
        """Buat satu baris ASS dengan tag karaoke \\k per kata."""
        parts: list[str] = []

        for i, word in enumerate(words):
            # \\k<durasi_centiseconds> — durasi kata dalam 1/100 detik
            prev_end = (words[i - 1].end - clip_start) if i > 0 else line_start
            curr_start = word.start - clip_start
            curr_end   = word.end   - clip_start

            # Gap sebelum kata ini
            gap_cs = max(0, int((curr_start - prev_end) * 100))
            if gap_cs > 0:
                parts.append(f"{{\\k{gap_cs}}} ")

            word_cs = max(1, int((curr_end - curr_start) * 100))
            parts.append(f"{{\\k{word_cs}}}{word.word}")

            if i < len(words) - 1:
                parts.append(" ")

        text = "".join(parts)
        return (
            f"Dialogue: 0,{self._ts(line_start)},{self._ts(line_end)},Karaoke,,0,0,0,,"
            f"{{\\an2}}{text}\n"
        )

    @staticmethod
    def _group_words_into_lines(
        words: list[Word], max_words_per_line: int = 8
    ) -> list[list[Word]]:
        """Bagi list kata menjadi baris-baris dengan panjang maksimum."""
        lines: list[list[Word]] = []
        current_line: list[Word] = []

        for word in words:
            current_line.append(word)
            # Mulai baris baru jika sudah maks kata ATAU ada jeda > 1.5 detik
            is_max = len(current_line) >= max_words_per_line
            is_gap = (
                len(current_line) > 1
                and (word.start - current_line[-2].end) > 1.5
            )
            if is_max or is_gap:
                lines.append(current_line)
                current_line = []

        if current_line:
            lines.append(current_line)

        return lines


# ---------------------------------------------------------------------------
# Video Editor (FFmpeg)
# ---------------------------------------------------------------------------

class VideoEditor:
    """
    Proses utama pemotongan dan rendering video klip:
    - Auto-crop 16:9 → 9:16 (center-crop atau split-screen)
    - Burn-in subtitle .ASS karaoke
    - Render final ke MP4 H.264
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._face_detector = FaceDetector()
        self._subtitle_gen  = ASSSubtitleGenerator()
        self._check_ffmpeg()

    def render_clip(self, config: RenderConfig) -> RenderResult:
        """
        Render satu klip vertikal 9:16 dari konfigurasi yang diberikan.

        Returns:
            RenderResult dengan path output dan status.
        """
        logger.info(
            "Rendering %s (%.1fs–%.1fs) …",
            config.clip_id, config.start_time, config.end_time,
        )

        output_path = self.output_dir / f"{config.clip_id}.mp4"
        tmp_dir = Path(tempfile.mkdtemp(prefix="clipper_"))

        try:
            # 1. Buat file subtitle .ASS
            ass_path = str(tmp_dir / f"{config.clip_id}.ass")
            self._subtitle_gen.generate(
                words=config.subtitle_words,
                headline=config.headline,
                clip_start=config.start_time,
                clip_end=config.end_time,
                output_path=ass_path,
            )

            # 2. Tentukan crop strategy
            crop_filter = self._build_crop_filter(
                video_path=config.input_video,
                start=config.start_time,
                end=config.end_time,
                speakers=config.speakers_detected,
            )

            # 3. Render via FFmpeg
            self._run_ffmpeg(
                input_path=config.input_video,
                output_path=str(output_path),
                start=config.start_time,
                end=config.end_time,
                crop_filter=crop_filter,
                ass_path=ass_path,
            )

            duration = config.end_time - config.start_time
            logger.info("✅ Klip selesai: %s (%.1fs)", output_path.name, duration)

            return RenderResult(
                clip_id=config.clip_id,
                output_path=str(output_path),
                duration=duration,
                success=True,
            )

        except subprocess.CalledProcessError as exc:
            err = f"FFmpeg error (exit {exc.returncode}): {exc.stderr}"
            logger.error(err)
            return RenderResult(
                clip_id=config.clip_id,
                output_path="",
                duration=0.0,
                success=False,
                error_message=err,
            )
        except Exception as exc:
            err = str(exc)
            logger.error("Render error untuk %s: %s", config.clip_id, err)
            return RenderResult(
                clip_id=config.clip_id,
                output_path="",
                duration=0.0,
                success=False,
                error_message=err,
            )
        finally:
            # Bersihkan temp files
            for f in tmp_dir.iterdir():
                f.unlink(missing_ok=True)
            tmp_dir.rmdir()

    def render_all(self, configs: list[RenderConfig]) -> list[RenderResult]:
        """Render semua klip secara berurutan."""
        results: list[RenderResult] = []
        for cfg in configs:
            results.append(self.render_clip(cfg))
        return results

    def close(self):
        self._face_detector.close()

    # ------------------------------------------------------------------
    # Crop Strategy
    # ------------------------------------------------------------------

    def _build_crop_filter(
        self,
        video_path: str,
        start: float,
        end: float,
        speakers: int,
    ) -> str:
        """
        Bangun FFmpeg filter string untuk crop 16:9 → 9:16.

        - 1 pembicara  : center-crop tracking berdasarkan wajah
        - 2 pembicara  : split-screen atas-bawah
        """
        src_w, src_h = self._get_video_dimensions(video_path)

        # Target: lebar = src_h * (9/16), tinggi = src_h
        target_w = int(src_h * 9 / 16)
        target_h = src_h

        if speakers >= 2:
            # Split-screen: crop kiri + kanan, tempel atas-bawah
            return self._split_screen_filter(src_w, src_h, target_w, target_h)

        # Single speaker: coba face tracking
        face_x = self._face_detector.get_dominant_face_x(video_path, start, end)

        if face_x is not None:
            # Pusatkan crop pada posisi wajah, clamp ke batas video
            crop_x = max(0, min(face_x - target_w // 2, src_w - target_w))
        else:
            # Fallback: center crop
            crop_x = (src_w - target_w) // 2

        return (
            f"crop={target_w}:{target_h}:{crop_x}:0,"
            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
        )

    @staticmethod
    def _split_screen_filter(src_w: int, src_h: int, target_w: int, target_h: int) -> str:
        """
        Split-screen atas-bawah: masing-masing bagian setengah tinggi.
        Pembicara 1 di atas, pembicara 2 di bawah.
        """
        half_h   = src_h // 2
        crop_x   = (src_w - target_w) // 2

        # [0:v] crop kiri-atas → [top]; crop kiri-bawah → [bot]; stack vertikal
        return (
            f"[0:v]split=2[v1][v2];"
            f"[v1]crop={target_w}:{half_h}:{crop_x}:0,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT // 2}[top];"
            f"[v2]crop={target_w}:{half_h}:{crop_x}:{half_h},scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT // 2}[bot];"
            f"[top][bot]vstack=inputs=2[outv]"
        )

    # ------------------------------------------------------------------
    # FFmpeg Runner
    # ------------------------------------------------------------------

    def _run_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        start: float,
        end: float,
        crop_filter: str,
        ass_path: str,
    ):
        duration = end - start
        is_split = "[outv]" in crop_filter

        # Escape path subtitle untuk FFmpeg (Windows/Linux safe)
        ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:")

        if is_split:
            # Filter graph kompleks untuk split-screen
            vf = f"{crop_filter};[outv]ass='{ass_escaped}'[final]"
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", input_path,
                "-t", str(duration),
                "-filter_complex", vf,
                "-map", "[final]",
                "-map", "0:a",
                "-c:v", "libx264",
                "-crf", str(VIDEO_CRF),
                "-preset", VIDEO_PRESET,
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                output_path,
            ]
        else:
            # Single filter chain
            vf = f"{crop_filter},ass='{ass_escaped}'"
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", input_path,
                "-t", str(duration),
                "-vf", vf,
                "-c:v", "libx264",
                "-crf", str(VIDEO_CRF),
                "-preset", VIDEO_PRESET,
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                output_path,
            ]

        logger.debug("FFmpeg cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )

        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "json",
                output_path,
            ],
            capture_output=True,
            text=True,
        )
        try:
            has_video_stream = bool(json.loads(probe.stdout).get("streams"))
        except (json.JSONDecodeError, AttributeError):
            has_video_stream = False

        if probe.returncode != 0 or not has_video_stream:
            Path(output_path).unlink(missing_ok=True)
            raise RuntimeError(
                "FFmpeg tidak menghasilkan stream video yang valid. "
                "Pastikan waktu klip berada di dalam durasi video sumber."
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_video_dimensions(video_path: str) -> tuple[int, int]:
        """Dapatkan lebar & tinggi video menggunakan ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])

    @staticmethod
    def _check_ffmpeg():
        """Verifikasi FFmpeg & FFprobe tersedia di PATH."""
        for tool in ("ffmpeg", "ffprobe"):
            result = subprocess.run(
                [tool, "-version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"'{tool}' tidak ditemukan. "
                    "Pastikan FFmpeg sudah terinstall dan ada di PATH."
                )
        logger.debug("FFmpeg dan FFprobe terdeteksi.")
