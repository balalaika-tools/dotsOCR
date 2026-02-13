from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO

import streamlit as st

from dotsOCR.core.batching import infer_pages_batched, join_results, run_async
from dotsOCR.core.images import (
    PageImage,
    load_image_from_bytes,
    load_pdf_pages_from_bytes,
    pdf_page_count_from_bytes,
    render_pdf_page_from_bytes,
)
from dotsOCR.core.md_transform import parse_json_from_response, response_to_markdown
from dotsOCR.core.prompts_legacy import dict_promptmode_to_prompt
from dotsOCR.core.vllm import VllmConfig


@dataclass(frozen=True)
class UiConfig:
    title: str
    vllm_base_url: str
    vllm_model: str
    vllm_api_key: str
    vllm_max_model_len: int


def load_ui_config() -> UiConfig:
    max_model_len_raw = os.environ.get("VLLM_MAX_MODEL_LEN", "12000")
    try:
        max_model_len = int(max_model_len_raw)
    except ValueError:
        max_model_len = 12000
    return UiConfig(
        title=os.environ.get("FRONTEND_TITLE", "DotsOCR"),
        vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1"),
        vllm_model=os.environ.get("VLLM_MODEL_NAME", "dotsOCR"),
        vllm_api_key=os.environ.get("VLLM_API_KEY", "0"),
        vllm_max_model_len=max_model_len,
    )


def _file_kind(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return "image"
    return "unknown"


MODE_LABELS: dict[str, str] = {
    "prompt_layout_all_en": "Layout+Text",
    "prompt_ocr": "Text",
}


def _mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def _prompt_output_kind(mode: str) -> str:
    if mode in {"prompt_layout_all_en", "prompt_layout_only_en"}:
        return "json"
    return "text"


def _available_prompt_modes() -> list[str]:
    # Keep only the two modes you requested.
    wanted = ["prompt_layout_all_en", "prompt_ocr"]
    return [m for m in wanted if m in dict_promptmode_to_prompt]


def _download_filename(*, base: str, page: int | None, ext: str) -> str:
    if page is None:
        return f"{base}.{ext}"
    return f"{base}_page-{page:03d}.{ext}"


def sidebar_controls(prompt_modes: list[str], default_mode: str) -> dict:
    st.sidebar.header("Inference")
    running = bool(st.session_state.get("is_inference_running", False))

    # Exactly 2 modes, fixed labels.
    label_to_mode = {
        "Layout+Text": "prompt_layout_all_en",
        "Text": "prompt_ocr",
    }
    selected_label = st.sidebar.radio("Mode", options=["Layout+Text", "Text"], index=0, disabled=running)
    selected_mode = label_to_mode[selected_label]
    target_dpi = st.sidebar.slider("PDF render DPI", min_value=96, max_value=300, value=200, step=12, disabled=running)
    max_pages = st.sidebar.number_input("Max pages (0 = all)", min_value=0, max_value=500, value=0, step=1, disabled=running)
    max_in_flight = st.sidebar.slider("Concurrency (auto-batching)", min_value=1, max_value=32, value=8, disabled=running)

    return {
        "selected_mode": selected_mode,
        "target_dpi": int(target_dpi),
        "max_pages": None if int(max_pages) == 0 else int(max_pages),
        "max_in_flight": int(max_in_flight),
    }


@st.cache_data(show_spinner=False)
def _cached_pdf_page_count(pdf_bytes: bytes) -> int:
    return pdf_page_count_from_bytes(pdf_bytes)


@st.cache_data(show_spinner=False)
def _cached_pdf_preview_png(pdf_bytes: bytes, *, page_index: int, target_dpi: int) -> bytes:
    img = render_pdf_page_from_bytes(pdf_bytes, page_index=page_index, target_dpi=target_dpi)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    cfg = load_ui_config()
    st.set_page_config(page_title=cfg.title, layout="wide")
    st.title(cfg.title)

    if "is_inference_running" not in st.session_state:
        st.session_state["is_inference_running"] = False
    if "auto_start_inference" not in st.session_state:
        st.session_state["auto_start_inference"] = False

    running = bool(st.session_state["is_inference_running"])
    prompt_modes = _available_prompt_modes()
    if not prompt_modes:
        st.error("No prompts available.")
        st.stop()

    controls = sidebar_controls(prompt_modes, prompt_modes[0])
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

    kind = _file_kind(uploaded.name)
    file_bytes = uploaded.getvalue()
    doc_key = f"{uploaded.name}:{len(file_bytes)}"

    if kind not in {"pdf", "image"}:
        st.error("Unsupported file type.")
        return

    # Preview navigation (fast): render only the selected page and cache it.
    if st.session_state.get("preview_doc_key") != doc_key:
        st.session_state["preview_doc_key"] = doc_key
        st.session_state["preview_page_index"] = 0

    preview_page_index = int(st.session_state.get("preview_page_index", 0))
    if kind == "pdf":
        page_count = _cached_pdf_page_count(file_bytes)
        if page_count <= 0:
            st.error("PDF had no pages.")
            return
        preview_page_index = max(0, min(preview_page_index, page_count - 1))
        st.session_state["preview_page_index"] = preview_page_index
        preview_png = _cached_pdf_preview_png(file_bytes, page_index=preview_page_index, target_dpi=controls["target_dpi"])
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
        st.write(f"Mode: {_mode_label(prompt_mode)}")
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
            "Generation: temperature=0.1, top_p=0.9 (hardcoded)"
        )

    if running:
        st.info("Inference is running in background. Please wait...")

    run_clicked = st.button("Run inference", type="primary", disabled=running)
    if run_clicked and not running:
        # Force one rerun with locked UI before starting heavy work.
        st.session_state["is_inference_running"] = True
        st.session_state["auto_start_inference"] = True
        st.rerun()

    if not st.session_state.get("auto_start_inference", False):
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

    progress = st.progress(0, text="Preparing requests...")
    status = st.empty()

    def _progress_update(phase: str, completed: int, total: int, last_page_index: int) -> None:
        if phase == "planning":
            pct = int((completed / max(total, 1)) * 30)
            progress.progress(pct, text=f"Planning token budget... {completed}/{total} pages")
            status.write(f"Planning completed for page {last_page_index + 1}")
            return

        pct = 30 + int((completed / max(total, 1)) * 70)
        progress.progress(pct, text=f"Running inference... {completed}/{total} pages")
        status.write(f"Inference completed for page {last_page_index + 1}")

    try:
        status.write("Running batched inference (vLLM auto-batching)...")
        results = run_async(
            infer_pages_batched(
                pages,
                cfg=vllm_cfg,
                prompt=prompt,
                max_in_flight=controls["max_in_flight"],
                progress_callback=_progress_update,
            )
        )
        progress.progress(100, text="Done.")
    finally:
        st.session_state["is_inference_running"] = False
        status.write("Inference finished.")

    # Precompute per-page artifacts so Combined works even if per-page UI is hidden.
    output_kind = _prompt_output_kind(prompt_mode)
    md_pages: list[str] = []
    json_pages: list[object] = []
    per_page: list[dict] = []
    for r in results:
        page_number = r.page_index + 1
        page_img = next((p.image for p in pages if p.page_index == r.page_index), None)
        item: dict = {"page_index": r.page_index, "page_number": page_number, "raw": r.content, "img": page_img}
        if output_kind == "json":
            try:
                parsed = parse_json_from_response(r.content)
                json_pages.append({"page": page_number, "result": parsed})
                item["parsed"] = parsed
                item["parsed_error"] = None
            except Exception as e:
                item["parsed"] = None
                item["parsed_error"] = str(e)

            if page_img is not None:
                try:
                    md = response_to_markdown(prompt_mode=prompt_mode, image=page_img, response_text=r.content)
                except Exception as e:
                    md = None
                    item["md_error"] = str(e)
                if md:
                    md_pages.append(md)
                    item["md"] = md
                    item["md_error"] = None
                else:
                    item["md"] = None
                    item["md_error"] = item.get("md_error") or "Markdown preview is available for `prompt_layout_all_en` only."
            else:
                item["md"] = None
                item["md_error"] = "No image available for markdown transform."
        per_page.append(item)

    st.subheader("Per-page outputs")
    tab_all, tab_pages = st.tabs(["All pages", "Pages"])
    with tab_all:
        st.caption("Per-page outputs are hidden. Open the `Pages` tab to view per-page results.")
    with tab_pages:
        for item in per_page:
            page_number = int(item["page_number"])
            with st.expander(f"Page {page_number}", expanded=(len(per_page) == 1)):
                if output_kind == "json":
                    tab_parsed, tab_md = st.tabs(["Parsed", "Markdown"])
                    with tab_parsed:
                        parsed = item.get("parsed")
                        parsed_error = item.get("parsed_error")
                        if parsed is not None:
                            st.download_button(
                                "Download (JSON)",
                                data=__import__("json").dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8"),
                                file_name=_download_filename(base=f"dotsocr_{prompt_mode}", page=page_number, ext="json"),
                                mime="application/json",
                            )
                            st.json(parsed)
                        else:
                            st.caption(f"JSON parse failed: {parsed_error}")
                    with tab_md:
                        md = item.get("md")
                        md_error = item.get("md_error")
                        if md:
                            st.download_button(
                                "Download (Markdown)",
                                data=md.encode("utf-8"),
                                file_name=_download_filename(base=f"dotsocr_{prompt_mode}", page=page_number, ext="md"),
                                mime="text/markdown",
                            )
                            st.markdown(md, unsafe_allow_html=True)
                        else:
                            st.caption(str(md_error))
                else:
                    st.download_button(
                        "Download (Text)",
                        data=str(item["raw"]).encode("utf-8"),
                        file_name=_download_filename(base=f"dotsocr_{prompt_mode}", page=page_number, ext="txt"),
                        mime="text/plain",
                    )
                    st.text_area("Text", str(item["raw"]), height=260)

    st.subheader("Combined")
    if output_kind == "json":
        tab_parsed, tab_md = st.tabs(["Parsed", "Markdown"])
        with tab_parsed:
            if json_pages:
                combined_json = json_pages[0]["result"] if len(json_pages) == 1 else json_pages
                if len(json_pages) > 1:
                    combined_bytes = (
                        "\n".join(__import__("json").dumps(item, ensure_ascii=False) for item in json_pages).strip() + "\n"
                    ).encode("utf-8")
                    combined_name = _download_filename(base=f"dotsocr_{prompt_mode}_combined", page=None, ext="jsonl")
                    combined_mime = "application/x-ndjson"
                else:
                    combined_bytes = __import__("json").dumps(combined_json, ensure_ascii=False, indent=2).encode("utf-8")
                    combined_name = _download_filename(base=f"dotsocr_{prompt_mode}_combined", page=None, ext="json")
                    combined_mime = "application/json"

                st.download_button(
                    "Download combined (Parsed)",
                    data=combined_bytes,
                    file_name=combined_name,
                    mime=combined_mime,
                )
            else:
                st.caption("No parsed JSON available.")

        with tab_md:
            if md_pages:
                # Lossless concat: keep per-page markdown intact; just separate pages cleanly.
                combined_md = ("\n\n---\n\n".join(md.rstrip() for md in md_pages)).strip() + "\n"
                st.download_button(
                    "Download combined (Markdown)",
                    data=combined_md.encode("utf-8"),
                    file_name=_download_filename(base=f"dotsocr_{prompt_mode}_combined", page=None, ext="md"),
                    mime="text/markdown",
                )
            else:
                st.caption("Markdown combined output is available for `prompt_layout_all_en` only.")
    else:
        combined_text = join_results(results, header=f"# {prompt_mode}")
        st.download_button(
            "Download combined (Text)",
            data=combined_text.encode("utf-8"),
            file_name=_download_filename(base=f"dotsocr_{prompt_mode}_combined", page=None, ext="txt"),
            mime="text/plain",
        )


if __name__ == "__main__":
    main()

