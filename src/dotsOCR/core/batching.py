from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Iterable

from openai import AsyncOpenAI, BadRequestError

from .images import PageImage
from .vllm import VllmConfig, build_async_client, build_vision_messages


@dataclass(frozen=True)
class PageResult:
    page_index: int
    content: str


def _safe_max_tokens(*, max_model_len: int, prompt_tokens: int, safety_margin: int = 32) -> int:
    # Avoid requesting more than available context window.
    return max(256, max_model_len - prompt_tokens - safety_margin)


async def _probe_prompt_tokens(
    client: AsyncOpenAI,
    *,
    cfg: VllmConfig,
    page: PageImage,
    prompt: str,
) -> int:
    """
    Probe request that always succeeds (max_tokens=1) to read back `usage.prompt_tokens`.
    This accounts for multimodal image tokens, so we can set `max_tokens` correctly
    without triggering a 400 and without heuristics.
    """
    messages = build_vision_messages(prompt=prompt, image=page.image)
    resp = await client.chat.completions.create(
        model=cfg.model,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
    )
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    if isinstance(prompt_tokens, int) and prompt_tokens > 0:
        return prompt_tokens
    # Fallback: if usage is missing, we can only be conservative.
    return max(1, int(cfg.max_model_len * 0.25))


async def infer_page(
    client: AsyncOpenAI,
    *,
    cfg: VllmConfig,
    page: PageImage,
    prompt: str,
    max_tokens: int,
) -> PageResult:
    messages = build_vision_messages(prompt=prompt, image=page.image)
    try:
        resp = await client.chat.completions.create(
            model=cfg.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=max_tokens,
        )
    except BadRequestError:
        # This should rarely happen now. Keep a small retry with a conservative clamp.
        clamped = max(256, int(max_tokens * 0.5))
        resp = await client.chat.completions.create(
            model=cfg.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=clamped,
        )
    return PageResult(page_index=page.page_index, content=resp.choices[0].message.content or "")


async def infer_pages_batched(
    pages: Iterable[PageImage],
    *,
    cfg: VllmConfig,
    prompt: str,
    max_in_flight: int = 8,
    progress_callback: Callable[[str, int, int, int], None] | None = None,
) -> list[PageResult]:
    """
    Fire multiple concurrent requests to leverage vLLM continuous batching.
    `max_in_flight` controls client-side concurrency (vLLM will auto-batch).
    """
    client = build_async_client(cfg)
    sem = asyncio.Semaphore(max_in_flight)

    pages_list = list(pages)

    async def _probe_one(p: PageImage) -> tuple[int, int]:
        async with sem:
            pt = await _probe_prompt_tokens(client, cfg=cfg, page=p, prompt=prompt)
            return p.page_index, pt

    # Phase 1: probe prompt tokens (no 400s).
    probe_tasks = [asyncio.create_task(_probe_one(p)) for p in pages_list]
    prompt_tokens_by_page: dict[int, int] = {}
    probe_total = len(probe_tasks)
    probe_done = 0
    for coro in asyncio.as_completed(probe_tasks):
        page_index, prompt_tokens = await coro
        prompt_tokens_by_page[page_index] = prompt_tokens
        probe_done += 1
        if progress_callback is not None:
            progress_callback("planning", probe_done, probe_total, page_index)

    async def _run_one(p: PageImage) -> PageResult:
        async with sem:
            pt = prompt_tokens_by_page.get(p.page_index, 1)
            mt = _safe_max_tokens(max_model_len=cfg.max_model_len, prompt_tokens=pt, safety_margin=32)
            return await infer_page(client, cfg=cfg, page=p, prompt=prompt, max_tokens=mt)

    # Phase 2: real inference, using safe max_tokens per page.
    tasks = [asyncio.create_task(_run_one(p)) for p in pages_list]
    results: list[PageResult] = []
    total = len(tasks)
    completed = 0
    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
        completed += 1
        if progress_callback is not None:
            progress_callback("inference", completed, total, res.page_index)
    results.sort(key=lambda r: r.page_index)
    return results


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Streamlit usually runs without a running loop, but keep a safe fallback.
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    return asyncio.run(coro)


def join_results(results: list[PageResult], *, header: str | None = None) -> str:
    parts: list[str] = []
    if header:
        parts.append(header.rstrip())
    for r in results:
        parts.append(f"\n\n---\n\n## Page {r.page_index + 1}\n\n{r.content.rstrip()}\n")
    return "".join(parts).strip() + "\n"

