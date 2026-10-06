from types import SimpleNamespace

import numpy as np
import pytest

import video_editor
from video_editor import ASSSubtitleGenerator, FaceDetector, VideoEditor


def test_face_detector_falls_back_when_mediapipe_solutions_is_missing(monkeypatch):
    monkeypatch.setattr(video_editor, "mp", SimpleNamespace())

    detector = FaceDetector()

    assert detector.detect_faces(np.zeros((32, 32, 3), dtype=np.uint8)) == []
    detector.close()


def test_ffmpeg_empty_output_is_rejected(monkeypatch, tmp_path):
    output_path = tmp_path / "empty.mp4"

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg":
            output_path.write_bytes(b"empty mp4")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout='{"streams": []}', stderr="")

    monkeypatch.setattr(video_editor.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="tidak menghasilkan stream video"):
        VideoEditor._run_ffmpeg(
            object(),
            input_path="input.mp4",
            output_path=str(output_path),
            start=120.0,
            end=150.0,
            crop_filter="crop=270:480:0:0,scale=1080:1920",
            ass_path=str(tmp_path / "subs.ass"),
        )

    assert not output_path.exists()


def test_subtitles_use_larger_readable_styles_and_wrap_long_headlines(tmp_path):
    subtitle_path = tmp_path / "style_test.ass"
    generator = ASSSubtitleGenerator()

    generator.generate(
        words=[],
        headline="KEPAKARAN DIGULINGKAN INFLUENCER MENGGUNCANG DUNIA",
        clip_start=0.0,
        clip_end=30.0,
        output_path=str(subtitle_path),
    )

    content = subtitle_path.read_text(encoding="utf-8")
    assert "Style: Karaoke,Arial,30," in content
    assert "Style: Headline,Arial,40," in content
    assert "&H80000000" in content
    assert r"\N" in content