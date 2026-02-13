from __future__ import annotations

import streamlit as st

from dotsOCR.core.batching import infer_pages_batched, run_async
from dotsOCR.core.images import PageImage
from dotsOCR.core.vllm import VllmConfig


def run_inference_with_progress(
    *,
    pages: list[PageImage],
    cfg: VllmConfig,
    prompt: str,
    max_in_flight: int,
) -> list:
    """
    Run inference and show a single progress bar that updates only when a page completes.

    Progress text format: "Scanning X/Y pages".
    """
    progress = st.progress(0, text="Preparing...")
    progress_state = {"last_completed": 0, "total": None}

    def _progress_update(phase: str, completed: int, total: int, last_page_index: int) -> None:
        if phase != "inference":
            return

        total_safe = max(int(total), 1)
        progress_state["total"] = total_safe

        completed_i = int(completed)
        if completed_i <= int(progress_state["last_completed"]):
            return
        progress_state["last_completed"] = completed_i

        pct = int((completed_i / total_safe) * 100)
        progress.progress(pct, text=f"Scanning {completed_i}/{total_safe} pages")

    results = run_async(
        infer_pages_batched(
            pages,
            cfg=cfg,
            prompt=prompt,
            max_in_flight=max_in_flight,
            progress_callback=_progress_update,
        )
    )

    total_done = int(progress_state.get("total") or max(len(pages), 1))
    progress.progress(100, text=f"Done. Scanned {total_done}/{total_done} pages")
    return results

