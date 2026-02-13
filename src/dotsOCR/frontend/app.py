from __future__ import annotations

from io import BytesIO

import streamlit as st

from dotsOCR.core.images import PageImage, load_image_from_bytes, load_pdf_pages_from_bytes
from dotsOCR.core.prompts_legacy import dict_promptmode_to_prompt
from dotsOCR.core.batching import join_results
from dotsOCR.core.vllm import VllmConfig

from dotsOCR.frontend.config import load_ui_config
from dotsOCR.frontend.controls import sidebar_controls
from dotsOCR.frontend.inference import run_inference_with_progress
from dotsOCR.frontend.pdf_cache import cached_pdf_page_count, cached_pdf_preview_png
from dotsOCR.frontend.results import build_per_page_artifacts, render_outputs
from dotsOCR.frontend.utils import available_prompt_modes, file_kind, mode_label

def main() -> None:
    cfg = load_ui_config()
    st.set_page_config(page_title=cfg.title, layout="wide")
    st.title(cfg.title)

    if "is_inference_running" not in st.session_state:
        st.session_state["is_inference_running"] = False
    if "auto_start_inference" not in st.session_state:
        st.session_state["auto_start_inference"] = False
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None

    running = bool(st.session_state["is_inference_running"])
    prompt_modes = available_prompt_modes(prompt_dict=dict_promptmode_to_prompt)
    if not prompt_modes:
        st.error("No prompts available.")
        st.stop()

    controls = sidebar_controls(running=running)
    prompt_mode = controls["selected_mode"]
    base_prompt = dict_promptmode_to_prompt[prompt_mode]

    prompt = base_prompt

    uploaded = st.file_uploader(
        "Upload an image or PDF",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        disabled=running,
        key="input_file",
    )
    col_a, col_b = st.columns([1, 1])

    if uploaded is None:
        with col_a:
            st.info("Upload a document to start.")
        with col_b:
            st.code(f"Using vLLM: {cfg.vllm_base_url}\nModel: {cfg.vllm_model}")
        return

    kind = file_kind(uploaded.name)
    file_bytes = uploaded.getvalue()
    doc_key = f"{uploaded.name}:{len(file_bytes)}"
    run_sig = {
        "doc_key": doc_key,
        "prompt_mode": prompt_mode,
        "target_dpi": int(controls["target_dpi"]),
        "max_pages": controls["max_pages"],
    }

    # If the user uploads a new document, drop the previous run cache.
    last_run = st.session_state.get("last_run")
    if isinstance(last_run, dict) and last_run.get("doc_key") != doc_key:
        st.session_state["last_run"] = None

    if kind not in {"pdf", "image"}:
        st.error("Unsupported file type.")
        return

    # Preview navigation (fast): render only the selected page and cache it.
    if st.session_state.get("preview_doc_key") != doc_key:
        st.session_state["preview_doc_key"] = doc_key
        st.session_state["preview_page_index"] = 0

    preview_page_index = int(st.session_state.get("preview_page_index", 0))
    if kind == "pdf":
        page_count = cached_pdf_page_count(file_bytes)
        if page_count <= 0:
            st.error("PDF had no pages.")
            return
        preview_page_index = max(0, min(preview_page_index, page_count - 1))
        st.session_state["preview_page_index"] = preview_page_index
        preview_png = cached_pdf_preview_png(file_bytes, page_index=preview_page_index, target_dpi=controls["target_dpi"])
    else:
        page_count = 1
        preview_page_index = 0
        st.session_state["preview_page_index"] = 0
        img = load_image_from_bytes(file_bytes)
        buf = BytesIO()
        img.save(buf, format="PNG")
        preview_png = buf.getvalue()

    with col_a:
        st.write(f"Pages: {page_count}")
        st.write(f"Mode: {mode_label(prompt_mode)}")
        if kind == "pdf":
            nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
            with nav_left:
                if st.button("← Prev", disabled=(running or preview_page_index == 0), use_container_width=True):
                    st.session_state["preview_page_index"] = preview_page_index - 1
                    st.rerun()
            with nav_mid:
                st.write(f"Preview page: {preview_page_index + 1} / {page_count}")
            with nav_right:
                if st.button("Next →", disabled=(running or preview_page_index == page_count - 1), use_container_width=True):
                    st.session_state["preview_page_index"] = preview_page_index + 1
                    st.rerun()
            st.image(preview_png, caption=f"Preview (page {preview_page_index + 1})", width="stretch")
        else:
            st.image(preview_png, caption="Preview", width="stretch")

    with col_b:
        st.code(
            f"vLLM: {cfg.vllm_base_url}\nModel: {cfg.vllm_model}\nVLLM_MAX_MODEL_LEN: {cfg.vllm_max_model_len}\n"
            "Generation: temperature=0.1, top_p=0.9"
        )

    if running:
        st.info("Inference is running in background. Please wait...")

    run_clicked = st.button("Run inference", type="primary", disabled=running)
    if run_clicked and not running:
        # Force one rerun with locked UI before starting heavy work.
        st.session_state["is_inference_running"] = True
        st.session_state["auto_start_inference"] = True
        st.rerun()

    # If we already have results for this exact input+settings, keep showing them.
    last_run = st.session_state.get("last_run")
    if not st.session_state.get("auto_start_inference", False):
        if isinstance(last_run, dict) and last_run.get("sig") == run_sig:
            render_outputs(
                prompt_mode=prompt_mode,
                per_page=list(last_run.get("per_page") or []),
                md_pages=list(last_run.get("md_pages") or []),
                json_pages=list(last_run.get("json_pages") or []),
                combined_text=last_run.get("combined_text"),
            )
        return
    st.session_state["auto_start_inference"] = False

    # Load pages only when running inference (can be slow for many pages).
    pages: list[PageImage]
    if kind == "pdf":
        pages = load_pdf_pages_from_bytes(file_bytes, target_dpi=controls["target_dpi"], max_pages=controls["max_pages"])
    else:
        pages = [PageImage(page_index=0, image=load_image_from_bytes(file_bytes))]

    vllm_cfg = VllmConfig(
        base_url=cfg.vllm_base_url,
        api_key=cfg.vllm_api_key,
        model=cfg.vllm_model,
        # hardcoded defaults from old repo demo_vllm.py / inference.py
        temperature=0.1,
        top_p=0.9,
        # Keep it high (maximal) but safe; batching layer will clamp/retry if needed.
        max_model_len=cfg.vllm_max_model_len,
    )

    try:
        results = run_inference_with_progress(
            pages=pages,
            cfg=vllm_cfg,
            prompt=prompt,
            max_in_flight=controls["max_in_flight"],
        )
    finally:
        st.session_state["is_inference_running"] = False

    # Precompute per-page artifacts so Combined works even if per-page UI is hidden.
    output_kind, per_page, md_pages, json_pages = build_per_page_artifacts(results=results, pages=pages, prompt_mode=prompt_mode)
    combined_text = None
    if output_kind == "text":
        combined_text = join_results(results, header=f"# {prompt_mode}")

    # Persist results so downloads/reruns don't wipe the UI.
    st.session_state["last_run"] = {
        "sig": run_sig,
        "doc_key": doc_key,
        "prompt_mode": prompt_mode,
        "target_dpi": int(controls["target_dpi"]),
        "max_pages": controls["max_pages"],
        "output_kind": output_kind,
        "per_page": per_page,
        "md_pages": md_pages,
        "json_pages": json_pages,
        "combined_text": combined_text,
    }
    render_outputs(
        prompt_mode=prompt_mode,
        per_page=per_page,
        md_pages=md_pages,
        json_pages=json_pages,
        combined_text=combined_text,
    )


if __name__ == "__main__":
    main()

