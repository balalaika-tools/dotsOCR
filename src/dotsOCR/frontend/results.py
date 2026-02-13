from __future__ import annotations

import json

import streamlit as st

from dotsOCR.core.images import PageImage
from dotsOCR.core.md_transform import parse_json_from_response, response_to_markdown, strip_base64_images_from_markdown

from .utils import download_filename, prompt_output_kind


def build_per_page_artifacts(
    *,
    results: list,
    pages: list[PageImage],
    prompt_mode: str,
) -> tuple[str, list[dict], list[str], list[object]]:
    """
    Build derived per-page outputs for rendering and downloads.
    Returns: (output_kind, per_page_items, md_pages, json_pages)
    """
    output_kind = prompt_output_kind(prompt_mode)
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

    return output_kind, per_page, md_pages, json_pages


def render_outputs(
    *,
    prompt_mode: str,
    per_page: list[dict],
    md_pages: list[str],
    json_pages: list[object],
    combined_text: str | None = None,
) -> None:
    output_kind = prompt_output_kind(prompt_mode)

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
                                data=json.dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8"),
                                file_name=download_filename(base=f"dotsocr_{prompt_mode}", page=page_number, ext="json"),
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
                                file_name=download_filename(base=f"dotsocr_{prompt_mode}", page=page_number, ext="md"),
                                mime="text/markdown",
                            )
                            st.markdown(md, unsafe_allow_html=True)
                        else:
                            st.caption(str(md_error))
                else:
                    st.download_button(
                        "Download (Text)",
                        data=str(item["raw"]).encode("utf-8"),
                        file_name=download_filename(base=f"dotsocr_{prompt_mode}", page=page_number, ext="txt"),
                        mime="text/plain",
                    )
                    st.text_area("Text", str(item["raw"]), height=260)

    st.subheader("Combined")
    if output_kind == "json":
        tab_parsed, tab_md = st.tabs(["Parsed", "Markdown"])
        with tab_parsed:
            if json_pages:
                combined_json = (
                    (json_pages[0]["result"] if isinstance(json_pages[0], dict) and "result" in json_pages[0] else json_pages[0])
                    if len(json_pages) == 1
                    else json_pages
                )
                if len(json_pages) > 1:
                    combined_bytes = ("\n".join(json.dumps(item, ensure_ascii=False) for item in json_pages).strip() + "\n").encode(
                        "utf-8"
                    )
                    combined_name = download_filename(base=f"dotsocr_{prompt_mode}_combined", page=None, ext="jsonl")
                    combined_mime = "application/x-ndjson"
                else:
                    combined_bytes = json.dumps(combined_json, ensure_ascii=False, indent=2).encode("utf-8")
                    combined_name = download_filename(base=f"dotsocr_{prompt_mode}_combined", page=None, ext="json")
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
                strip_images = st.checkbox("Remove embedded images (base64) from combined Markdown", value=False)
                pages_for_join = (
                    [strip_base64_images_from_markdown(md).rstrip() for md in md_pages] if strip_images else [md.rstrip() for md in md_pages]
                )
                combined_md = ("\n\n---\n\n".join(p for p in pages_for_join if p)).strip() + "\n"
                st.download_button(
                    "Download combined (Markdown)",
                    data=combined_md.encode("utf-8"),
                    file_name=download_filename(base=f"dotsocr_{prompt_mode}_combined", page=None, ext="md"),
                    mime="text/markdown",
                )
            else:
                st.caption("Markdown combined output is available for `prompt_layout_all_en` only.")
    else:
        if not combined_text:
            combined_text = "\n".join(str(item.get("raw", "")).rstrip() for item in per_page).strip() + "\n"
        st.download_button(
            "Download combined (Text)",
            data=combined_text.encode("utf-8"),
            file_name=download_filename(base=f"dotsocr_{prompt_mode}_combined", page=None, ext="txt"),
            mime="text/plain",
        )

