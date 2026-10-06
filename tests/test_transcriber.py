import httpx

import transcriber
from transcriber import Transcriber


def test_transcriber_retries_http_408(monkeypatch, tmp_path):
    request = httpx.Request("POST", "https://api.deepgram.com/v1/listen")
    responses = [
        httpx.Response(408, text="upstream timeout", request=request),
        httpx.Response(200, json={"ok": True}, request=request),
    ]
    request_count = 0
    sleep_calls = []

    class FakeClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def post(self, *args, **kwargs):
            nonlocal request_count
            request_count += 1
            return responses.pop(0)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav-data")
    monkeypatch.setattr(transcriber.httpx, "Client", FakeClient)
    monkeypatch.setattr(transcriber.time, "sleep", sleep_calls.append)
    monkeypatch.setattr(Transcriber, "_parse_response", lambda self, payload: payload)

    result = Transcriber(api_key="test-key")._transcribe_audio_file(audio_path)

    assert result == {"ok": True}
    assert request_count == 2
    assert sleep_calls == [5]


def test_parse_response_handles_rest_word_dictionaries():
    response = {
        "metadata": {"duration": 12.5},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Halo dunia",
                            "words": [
                                {
                                    "word": "Halo",
                                    "start": 0.1,
                                    "end": 0.5,
                                    "confidence": 0.99,
                                    "speaker": 0,
                                },
                                {
                                    "word": "dunia",
                                    "start": 0.6,
                                    "end": 1.0,
                                    "confidence": 0.98,
                                    "speaker": 1,
                                },
                            ],
                        }
                    ]
                }
            ]
        },
    }

    result = Transcriber(api_key="test-key")._parse_response(response)

    assert result.full_text == "Halo dunia"
    assert result.duration_seconds == 12.5
    assert result.speakers_detected == 2
    assert result.words[0].word == "Halo"
    assert result.words[0].start == 0.1
    assert result.words[1].speaker == 1