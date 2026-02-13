from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Iterable

import fitz  # PyMuPDF
from PIL import Image


@dataclass(frozen=True)
class PageImage:
    page_index: int
    image: Image.Image


def load_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB")


def _render_page_to_image(page: fitz.Page, target_dpi: int) -> Image.Image:
    mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    # Fallback if too large (avoid huge RAM spikes)
    if pm.width > 4500 or pm.height > 4500:
        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    return Image.frombytes("RGB", (pm.width, pm.height), pm.samples)


def load_pdf_pages_from_bytes(
    data: bytes,
    *,
    target_dpi: int = 200,
    max_pages: int | None = None,
) -> list[PageImage]:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        total = doc.page_count
        limit = total if max_pages is None else min(total, max_pages)
        out: list[PageImage] = []
        for i in range(limit):
            page = doc.load_page(i)
            out.append(PageImage(page_index=i, image=_render_page_to_image(page, target_dpi)))
        return out
    finally:
        doc.close()


def pdf_page_count_from_bytes(data: bytes) -> int:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def render_pdf_page_from_bytes(data: bytes, *, page_index: int, target_dpi: int = 200) -> Image.Image:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page_index = max(0, min(int(page_index), doc.page_count - 1))
        page = doc.load_page(page_index)
        return _render_page_to_image(page, target_dpi)
    finally:
        doc.close()


def pil_to_data_url_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def clamp_pages(pages: Iterable[PageImage], *, start: int = 0, end_inclusive: int | None = None) -> list[PageImage]:
    out: list[PageImage] = []
    for p in pages:
        if p.page_index < start:
            continue
        if end_inclusive is not None and p.page_index > end_inclusive:
            continue
        out.append(p)
    return out

