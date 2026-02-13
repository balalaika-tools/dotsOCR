from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAI

from .images import pil_to_data_url_png


DOTS_OCR_PROMPT_PREFIX = "<|img|><|imgpad|><|endofimg|>"


@dataclass(frozen=True)
class VllmConfig:
    base_url: str  # e.g. http://vllm:8000/v1
    api_key: str = "0"
    model: str = "dotsOCR"
    temperature: float = 0.1
    top_p: float = 0.9
    # vLLM server-side max context length (`--max-model-len`).
    max_model_len: int = 12000


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def build_client(cfg: VllmConfig) -> OpenAI:
    return OpenAI(api_key=cfg.api_key, base_url=_normalize_base_url(cfg.base_url))


def build_async_client(cfg: VllmConfig) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=cfg.api_key, base_url=_normalize_base_url(cfg.base_url))


def build_vision_messages(*, prompt: str, image) -> list[dict[str, Any]]:
    # vLLM OpenAI-compatible multimodal format.
    # Prefix avoids vLLM inserting a newline before the prompt in some setups.
    text = f"{DOTS_OCR_PROMPT_PREFIX}{prompt}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": pil_to_data_url_png(image)}},
                {"type": "text", "text": text},
            ],
        }
    ]

