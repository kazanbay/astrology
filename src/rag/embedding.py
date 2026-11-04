"""Shared BGE-M3 model configuration for indexing and retrieval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip() or "BAAI/bge-m3"
_REQUESTED_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower() or "auto"
if _REQUESTED_DEVICE not in {"auto", "cpu", "cuda"}:
    raise ValueError("EMBEDDING_DEVICE must be one of: auto, cpu, cuda")

# Use CUDA when PyTorch can access it; half precision leaves room on 3 GB cards.
DEVICE = "cuda" if _REQUESTED_DEVICE == "auto" and torch.cuda.is_available() else _REQUESTED_DEVICE
if DEVICE == "auto":
    DEVICE = "cpu"


def configure_torch_threads() -> None:
    """Configure CPU thread allocation if running on CPU."""
    if DEVICE == "cpu":
        # Берём количество ядер системы (или 1, если os.cpu_count() вернул None)
        total_cores = os.cpu_count() or 1

        # Для CPU оптимально задействовать основные физические/логические потоки
        torch.set_num_threads(total_cores)

        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Нельзя менять inter-op настройки, если PyTorch уже запустил параллельные задачи
            pass


def load_bge_model():
    configure_torch_threads()
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("EMBEDDING_DEVICE=cuda, but PyTorch cannot access a CUDA device")
    from sentence_transformers import SentenceTransformer

    print(f"Loading {MODEL_NAME} on {DEVICE}", flush=True)

    # BGE-M3's repository provides PyTorch bin weights, so don't require safetensors.
    model_kwargs: dict[str, Any] = {"use_safetensors": False}
    if DEVICE == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    return SentenceTransformer(MODEL_NAME, device=DEVICE, model_kwargs=model_kwargs)
