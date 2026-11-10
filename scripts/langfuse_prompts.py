"""Synchronize the local prompt constants with Langfuse.

Modes:
* ``check`` (default) - compare the local prompts with the ``production``
  versions in Langfuse; read-only report.
* ``bootstrap`` - create the prompts that do not exist yet with the
  ``production`` label (first-time setup).
* ``update`` - for prompts whose production text differs from the local
  constants, create a new version and move ``production`` to it. Use this
  only when you deliberately want to publish the local text; otherwise edit
  the prompt in the Langfuse UI.

Usage:
    python scripts/langfuse_prompts.py            # check
    python scripts/langfuse_prompts.py bootstrap
    python scripts/langfuse_prompts.py update
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

import llm_prompts  # noqa: E402
from generator.prompts import ASTROLOGER_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402
from rag.generator import SYSTEM_PROMPT  # noqa: E402


@dataclass(frozen=True)
class PromptSpec:
    """One prompt: its Langfuse name, type and the local source of truth."""

    name: str
    kind: str  # "chat" | "text"
    local: Any  # list[dict[str, str]] for chat, str for text


def local_interpretation_chat() -> list[dict[str, str]]:
    """The local natal-chart prompt as a Langfuse chat template.

    ``USER_PROMPT_TEMPLATE`` uses Python-style ``{chart_json}`` placeholders;
    Langfuse templates use the double-brace form.
    """
    user_template = USER_PROMPT_TEMPLATE.replace("{chart_json}", "{{chart_json}}").replace(
        "{chunks}", "{{chunks}}"
    )
    return [
        {"role": "system", "content": ASTROLOGER_SYSTEM_PROMPT},
        {"role": "user", "content": user_template},
    ]


def local_specs() -> list[PromptSpec]:
    return [
        PromptSpec(
            llm_prompts.prompt_name(llm_prompts.INTERPRETATION_PROMPT_NAME),
            "chat",
            local_interpretation_chat(),
        ),
        PromptSpec(llm_prompts.prompt_name(llm_prompts.QA_PROMPT_NAME), "text", SYSTEM_PROMPT),
    ]


def normalize(spec: PromptSpec) -> Any:
    """Shape of the local content, matching what ``get_prompt`` returns."""
    if spec.kind == "chat":
        return [(m["role"], m["content"]) for m in spec.local]
    return spec.local


def fetch_production(client: Any, spec: PromptSpec) -> Any | None:
    """Return the production content, or ``None`` when the prompt is missing."""
    try:
        prompt = client.get_prompt(spec.name, label=llm_prompts.prompt_label(), type=spec.kind)
    except Exception as exc:  # noqa: BLE001 - distinguish "missing" from "unreachable"
        status = getattr(exc, "status_code", None)
        if status == 404 or "404" in str(exc):
            return None
        raise
    if spec.kind == "chat":
        return [
            (m["role"], m["content"]) for m in getattr(prompt, "prompt", []) if m.get("type") == "message"
        ]
    return getattr(prompt, "prompt", None)


def create_production(client: Any, spec: PromptSpec) -> int:
    """Create a new version with the production label; return its number."""
    created = client.create_prompt(
        name=spec.name,
        prompt=spec.local,
        type=spec.kind,
        labels=[llm_prompts.prompt_label()],
        commit_message="sync from local constants (scripts/langfuse_prompts.py)",
    )
    return int(getattr(created, "version", 0))


def connect() -> Any:
    try:
        from langfuse import get_client
    except ImportError:
        print("Пакет langfuse не установлен в текущем окружении.")
        raise SystemExit(2)
    try:
        return get_client()
    except Exception as exc:  # noqa: BLE001 - configuration problems surface here
        print(f"Не удалось создать клиент Langfuse: {exc}")
        print("Проверьте LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL в .env")
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync local prompts with Langfuse")
    parser.add_argument("mode", nargs="?", default="check", choices=["check", "bootstrap", "update"])
    args = parser.parse_args()

    client = connect()
    specs = local_specs()
    failed = False

    for spec in specs:
        try:
            current = fetch_production(client, spec)
        except Exception as exc:  # noqa: BLE001 - report and continue with other prompts
            print(f"[ОШИБКА] {spec.name}: Langfuse недоступен ({exc})")
            failed = True
            continue

        if current is None:
            if args.mode == "bootstrap":
                version = create_production(client, spec)
                print(f"[СОЗДАН] {spec.name}: версия {version} (label: {llm_prompts.prompt_label()})")
            else:
                print(f"[НЕТ В LANGFUSE] {spec.name}: создайте через bootstrap или в UI Langfuse")
            continue

        if current == normalize(spec):
            print(f"[СИНХРОННО] {spec.name}: production совпадает с локальной константой")
            continue

        if args.mode == "update":
            version = create_production(client, spec)
            print(f"[ОБНОВЛЕН] {spec.name}: новая версия {version} (label: {llm_prompts.prompt_label()})")
        else:
            print(f"[РАЗЛИЧИЕ] {spec.name}: production в Langfuse отличается от локальной константы.")
            print("           Редактируйте промпт в UI Langfuse или запустите `update`,")
            print("           чтобы опубликовать локальный текст новой версией.")
            if args.mode == "check":
                failed = True

    if args.mode == "check" and not failed:
        print("\nПроверка завершена: расхождений нет.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
