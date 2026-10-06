from clip_selector import ClipInstruction, ClipSelectionResult
from main import AIClipperPipeline


def test_cache_key_changes_when_input_video_changes(tmp_path):
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    first_video.write_bytes(b"video-one")
    second_video.write_bytes(b"video-two")

    assert AIClipperPipeline._cache_key(first_video) != AIClipperPipeline._cache_key(second_video)
    assert AIClipperPipeline._cache_key(first_video) == AIClipperPipeline._cache_key(first_video)


def test_cached_selection_round_trip_ignores_computed_duration():
    selection = ClipSelectionResult(
        total_clips_found=1,
        clips=[
            ClipInstruction(
                clip_id="clip_001",
                headline="HOOK",
                start_time=10.0,
                end_time=40.0,
                virality_score=0.9,
                hook_statement="Kalimat pembuka",
                reason="Momen menarik",
            )
        ],
        provider_used="groq",
    )

    restored = AIClipperPipeline._dict_to_selection(selection.to_dict())

    assert restored.clips[0].duration == 30.0
    assert restored.clips[0].headline == "HOOK"
    assert restored.provider_used == "groq"