from __future__ import annotations


MODE_LABELS: dict[str, str] = {
    "prompt_layout_all_en": "Layout+Text",
    "prompt_ocr": "Text",
}


def file_kind(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return "image"
    return "unknown"


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def prompt_output_kind(mode: str) -> str:
    if mode in {"prompt_layout_all_en", "prompt_layout_only_en"}:
        return "json"
    return "text"


def available_prompt_modes(*, prompt_dict: dict[str, str]) -> list[str]:
    # Keep only the two modes requested.
    wanted = ["prompt_layout_all_en", "prompt_ocr"]
    return [m for m in wanted if m in prompt_dict]


def download_filename(*, base: str, page: int | None, ext: str) -> str:
    if page is None:
        return f"{base}.{ext}"
    return f"{base}_page-{page:03d}.{ext}"

