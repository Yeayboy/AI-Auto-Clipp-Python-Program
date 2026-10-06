import os

from clip_selector import ClipSelector, GROQ_MODEL, LLMProvider, SYSTEM_PROMPT
from transcriber import TranscriptResult


def test_llm_provider_has_groq():
    assert LLMProvider.GROQ.value == "groq"
    assert GROQ_MODEL == "openai/gpt-oss-20b"


def test_clip_selector_supports_groq_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    selector = ClipSelector(provider=LLMProvider.GROQ)
    assert selector.provider == LLMProvider.GROQ
    assert hasattr(selector, "_groq")


def test_headline_prompt_and_parser_keep_titles_grounded_and_concise():
    assert "benar-benar didukung isi klip" in SYSTEM_PROMPT
    assert "jangan menebak" in SYSTEM_PROMPT

    selector = ClipSelector.__new__(ClipSelector)
    transcript = TranscriptResult(full_text="", duration_seconds=30)
    response = (
        '{"clips":[{"start_time":0,"end_time":30,'
        '"headline":"ayam goreng enak banget untuk dicoba di sini ya"}]}'
    )

    result = selector._parse_llm_response(response, transcript)

    assert result.clips[0].headline == "AYAM GORENG ENAK BANGET UNTUK DICOBA DI SINI"


def test_multimodal_analyzer_uses_new_google_genai_client(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    from multimodal_analyzer import MultimodalAnalyzer

    analyzer = MultimodalAnalyzer()
    assert hasattr(analyzer, "client")
    assert analyzer.client is not None
