from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Iterable

import fitz  # PyMuPDF
from PIL import Image

INFERENCE_MAX_IMAGE_SIDE = 2048
IMAGE_FACTOR = 28
MIN_PIXELS = 3136
MAX_PIXELS = INFERENCE_MAX_IMAGE_SIDE * INFERENCE_MAX_IMAGE_SIDE


@dataclass(frozen=True)
class PageImage:
    page_index: int
    image: Image.Image


def _to_rgb(img: Image.Image) -> Image.Image:
    # Preserve readability for transparent images by compositing on white.
    if img.mode == "RGBA":
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.split()[3])
        return white
    return img.convert("RGB")


def load_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    return _to_rgb(img)


def resize_to_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale image so its longest side is <= max_side."""
    max_side_i = max(1, int(max_side))
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side_i:
        return img
    ratio = max_side_i / float(longest)
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    *,
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Keep aspect ratio and force dimensions to a model-friendly grid (factor).
    """
    if min(height, width) <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((h_bar * w_bar) / max_pixels)
            h_bar = max(factor, _floor_by_factor(h_bar / beta, factor))
            w_bar = max(factor, _floor_by_factor(w_bar / beta, factor))

    return int(h_bar), int(w_bar)


def resize_for_model(img: Image.Image) -> Image.Image:
    """
    Apply model-aware resize constraints (factor grid + min/max pixels).
    """
    h, w = img.height, img.width
    new_h, new_w = smart_resize(height=h, width=w, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    if new_h == h and new_w == w:
        return img
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _render_page_to_image(page: fitz.Page, target_dpi: int) -> Image.Image:
    # For inference we want deterministic rendering at the requested DPI.
    # We'll only fall back if rendering at that DPI fails (e.g. memory constraints).
    dpi = int(target_dpi)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    try:
        pm = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
    except (MemoryError, RuntimeError):
        # Fallback: progressively reduce DPI to avoid crashes.
        return _render_page_to_image_capped(page, target_dpi=dpi, max_side=4500, min_dpi=96)


def _render_page_to_image_capped(page: fitz.Page, *, target_dpi: int, max_side: int = 4500, min_dpi: int = 96) -> Image.Image:
    """
    Render a PDF page while capping output dimensions to avoid huge RAM spikes.
    Used as a safety fallback (and for previews).
    """
    dpi = int(target_dpi)
    while True:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pm = page.get_pixmap(matrix=mat, alpha=False)
        if pm.width <= max_side and pm.height <= max_side:
            return Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        if dpi <= min_dpi:
            return Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        dpi = max(min_dpi, int(dpi * 0.8))


def load_pdf_pages_from_bytes(
    data: bytes,
    *,
    target_dpi: int = 200,
    max_pages: int | None = None,
    max_side: int = INFERENCE_MAX_IMAGE_SIDE,
) -> list[PageImage]:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        total = doc.page_count
        limit = total if max_pages is None else min(total, max_pages)
        out: list[PageImage] = []
        for i in range(limit):
            page = doc.load_page(i)
            img = _render_page_to_image(page, target_dpi)
            img = resize_to_max_side(img, max_side)
            img = resize_for_model(img)
            out.append(PageImage(page_index=i, image=img))
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
        # Preview path: keep it safe to avoid huge UI renders.
        return _render_page_to_image_capped(page, target_dpi=int(target_dpi), max_side=4500, min_dpi=96)
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

