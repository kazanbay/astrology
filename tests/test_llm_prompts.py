"""Unit tests for the Langfuse prompt hub (offline, fake clients)."""

import asyncio
from types import SimpleNamespace

import pytest

import llm_prompts
from generator.prompts import PROMPT_VERSION
from generator.report import ChartReportService
from generator.synthesizer import synthesize_chart_report
from llm_prompts import INTERPRETATION_PROMPT_NAME, QA_PROMPT_NAME, resolve, resolve_chat

CHAT_ALLOWED = ("chart_json", "chunks")


def _enable_langfuse(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PROMPTS_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")


class FakeTextPrompt:
    def __init__(self, text="Промпт из Langfuse", version=4):
        self.prompt = text
        self.version = version
        self.name = "astrology_qa"


class FakeChatPrompt:
    version = 2
    prompt = [
        {"type": "message", "role": "system", "content": "Система из Langfuse"},
        {"type": "message", "role": "user", "content": "Карта: {{chart_json}} Фрагменты: {{chunks}}"},
    ]

    def compile(self, **kwargs):
        return [
            {"role": "system", "content": "Система из Langfuse"},
            {"role": "user", "content": f"Карта: {kwargs['chart_json']} Фрагменты: {kwargs['chunks']}"},
        ]


def _fake_client(monkeypatch, prompt):
    calls = []

    class FakeLangfuse:
        def get_prompt(self, name, **kwargs):
            calls.append((name, kwargs))
            return prompt

    monkeypatch.setattr("langfuse.get_client", lambda: FakeLangfuse())
    return calls


def test_text_prompt_falls_back_without_credentials(monkeypatch):
    _enable_langfuse(monkeypatch)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    resolution = resolve(QA_PROMPT_NAME)

    assert resolution.source == "local"
    assert resolution.text is None
    assert resolution.version == "local"


def test_text_prompt_falls_back_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PROMPTS_ENABLED", "false")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(
        "langfuse.get_client",
        lambda: (_ for _ in ()).throw(AssertionError("get_client must not be called")),
    )

    resolution = resolve(QA_PROMPT_NAME)

    assert resolution.source == "local"


def test_text_prompt_uses_langfuse(monkeypatch):
    _enable_langfuse(monkeypatch)
    calls = _fake_client(monkeypatch, FakeTextPrompt())

    resolution = resolve(QA_PROMPT_NAME)

    assert resolution.source == "langfuse"
    assert resolution.text == "Промпт из Langfuse"
    assert resolution.version == "langfuse:astrology_qa:v4"
    assert calls and calls[0][0] == "astrology_qa"
    assert calls[0][1]["type"] == "text"
    assert calls[0][1]["label"] == "production"


def test_text_prompt_falls_back_on_client_error(monkeypatch):
    _enable_langfuse(monkeypatch)

    class Broken:
        def get_prompt(self, name, **kwargs):
            raise RuntimeError("network down")

    monkeypatch.setattr("langfuse.get_client", lambda: Broken())

    resolution = resolve(QA_PROMPT_NAME)

    assert resolution.source == "local"
    assert resolution.text is None


def test_text_prompt_rejects_empty_text(monkeypatch):
    _enable_langfuse(monkeypatch)
    _fake_client(monkeypatch, FakeTextPrompt(text="   "))

    resolution = resolve(QA_PROMPT_NAME)

    assert resolution.source == "local"


def test_chat_prompt_uses_langfuse_and_compiles(monkeypatch):
    _enable_langfuse(monkeypatch)
    calls = _fake_client(monkeypatch, FakeChatPrompt())

    resolution = resolve_chat(
        INTERPRETATION_PROMPT_NAME, allowed_variables=CHAT_ALLOWED, local_version="v1.3"
    )
    messages = resolution.compile(chart_json="{}JSON{}", chunks="[]CHUNKS[]")

    assert resolution.source == "langfuse"
    assert resolution.version == "langfuse:astrology_interpretation:v2"
    assert messages == [
        {"role": "system", "content": "Система из Langfuse"},
        {"role": "user", "content": "Карта: {}JSON{} Фрагменты: []CHUNKS[]"},
    ]
    assert calls[0][0] == "astrology_interpretation"
    assert calls[0][1]["type"] == "chat"
    assert calls[0][1]["label"] == "production"


def test_chat_prompt_rejects_unknown_variable(monkeypatch):
    _enable_langfuse(monkeypatch)

    class Prompt:
        version = 2
        prompt = [{"type": "message", "role": "user", "content": "{{chart_json}} {{user_name}}"}]

        def compile(self, **kwargs):
            raise AssertionError("must not compile a rejected template")

    _fake_client(monkeypatch, Prompt())

    resolution = resolve_chat(
        INTERPRETATION_PROMPT_NAME, allowed_variables=CHAT_ALLOWED, local_version="v1.3"
    )

    assert resolution.source == "local"
    assert resolution.version == "v1.3"
    assert resolution.client is None


def test_chat_prompt_rejects_missing_variable(monkeypatch):
    _enable_langfuse(monkeypatch)

    class Prompt:
        version = 2
        prompt = [{"type": "message", "role": "user", "content": "{{chart_json}}"}]

        def compile(self, **kwargs):
            raise AssertionError("must not compile a rejected template")

    _fake_client(monkeypatch, Prompt())

    resolution = resolve_chat(
        INTERPRETATION_PROMPT_NAME, allowed_variables=CHAT_ALLOWED, local_version="v1.3"
    )

    assert resolution.source == "local"


def test_chat_prompt_rejects_placeholder_entry(monkeypatch):
    _enable_langfuse(monkeypatch)

    class Prompt:
        version = 2
        prompt = [
            {"type": "message", "role": "system", "content": "Система"},
            {"type": "placeholder", "name": "history"},
        ]

        def compile(self, **kwargs):
            raise AssertionError("must not compile a rejected template")

    _fake_client(monkeypatch, Prompt())

    resolution = resolve_chat(
        INTERPRETATION_PROMPT_NAME, allowed_variables=CHAT_ALLOWED, local_version="v1.3"
    )

    assert resolution.source == "local"


def test_resolution_is_memoized(monkeypatch):
    _enable_langfuse(monkeypatch)
    calls = _fake_client(monkeypatch, FakeChatPrompt())

    resolve_chat(INTERPRETATION_PROMPT_NAME, allowed_variables=CHAT_ALLOWED, local_version="v1.3")
    resolve_chat(INTERPRETATION_PROMPT_NAME, allowed_variables=CHAT_ALLOWED, local_version="v1.3")
    assert len(calls) == 1

    llm_prompts.reset_cache()
    resolve_chat(INTERPRETATION_PROMPT_NAME, allowed_variables=CHAT_ALLOWED, local_version="v1.3")
    assert len(calls) == 2


def test_prompt_name_env_override(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PROMPT_NAME_QA", "custom_qa")

    assert llm_prompts.prompt_name(QA_PROMPT_NAME) == "custom_qa"
    assert llm_prompts.prompt_name(INTERPRETATION_PROMPT_NAME) == "astrology_interpretation"


def test_prompt_label_env_override(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PROMPT_LABEL", "staging")

    assert llm_prompts.prompt_label() == "staging"


def test_synthesizer_uses_langfuse_messages(monkeypatch):
    _enable_langfuse(monkeypatch)
    _fake_client(monkeypatch, FakeChatPrompt())

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Разбор"))]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    text = asyncio.run(synthesize_chart_report({"Sun": {"sign": "Leo"}}, [], client=client))

    assert text == "Разбор"
    messages = completions.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "Система из Langfuse"}
    assert "Sun" in messages[1]["content"]


def test_report_service_takes_version_from_langfuse(monkeypatch):
    _enable_langfuse(monkeypatch)
    _fake_client(monkeypatch, FakeChatPrompt())

    service = ChartReportService(SimpleNamespace())

    assert service.prompt_version == "langfuse:astrology_interpretation:v2"


def test_report_service_falls_back_to_local_version(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PROMPTS_ENABLED", "false")

    service = ChartReportService(SimpleNamespace())

    assert service.prompt_version == PROMPT_VERSION
