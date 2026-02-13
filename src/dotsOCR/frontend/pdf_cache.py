from __future__ import annotations

from io import BytesIO

import streamlit as st

from dotsOCR.core.images import pdf_page_count_from_bytes, render_pdf_page_from_bytes


# Cache only a small window of PDF previews to keep navigation fast without growing memory unbounded.
# Entries expire after 5 minutes and we keep up to 30 entries.
@st.cache_data(show_spinner=False, ttl=300, max_entries=30)
def cached_pdf_page_count(pdf_bytes: bytes) -> int:
    return pdf_page_count_from_bytes(pdf_bytes)


@st.cache_data(show_spinner=False, ttl=300, max_entries=30)
def cached_pdf_preview_png(pdf_bytes: bytes, *, page_index: int, target_dpi: int) -> bytes:
    img = render_pdf_page_from_bytes(pdf_bytes, page_index=page_index, target_dpi=target_dpi)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

