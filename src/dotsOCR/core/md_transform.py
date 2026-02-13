from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image

from .images import pil_to_data_url_png


def has_latex_markdown(text: str) -> bool:
    if not isinstance(text, str):
        return False
    latex_patterns = [
        r"\$\$.*?\$\$",
        r"\$[^$\n]+?\$",
        r"\\begin\{.*?\}.*?\\end\{.*?\}",
        r"\\[a-zA-Z]+\{.*?\}",
        r"\\[a-zA-Z]+",
        r"\\\[.*?\\\]",
        r"\\\(.*?\\\)",
    ]
    return any(re.search(p, text, re.DOTALL) for p in latex_patterns)


def clean_latex_preamble(latex_text: str) -> str:
    patterns = [
        r"\\documentclass\{[^}]+\}",
        r"\\usepackage\{[^}]+\}",
        r"\\usepackage\[[^\]]*\]\{[^}]+\}",
        r"\\begin\{document\}",
        r"\\end\{document\}",
    ]
    cleaned_text = latex_text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text


def get_formula_in_markdown(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("$$") and text.endswith("$$"):
        text_new = text[2:-2].strip()
        if "$" not in text_new:
            return f"$$\n{text_new}\n$$"
        return text

    if text.startswith("\\[") and text.endswith("\\]"):
        inner_content = text[2:-2].strip()
        return f"$$\n{inner_content}\n$$"

    if len(re.findall(r".*\\\[.*\\\].*", text)) > 0:
        return text

    pattern = r"\$([^$]+)\$"
    if len(re.findall(pattern, text)) > 0:
        return text

    if not has_latex_markdown(text):
        return text

    if "usepackage" in text:
        text = clean_latex_preamble(text)

    if text and text[0] == "`" and text[-1] == "`":
        text = text[1:-1]

    return f"$$\n{text}\n$$"


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if text[:2] == "`$" and text[-2:] == "$`":
        text = text[1:-1]
    return text


def layoutjson2md(image: Image.Image, cells: list[dict], text_key: str = "text", no_page_hf: bool = False) -> str:
    text_items: list[str] = []

    for cell in cells:
        x1, y1, x2, y2 = [int(coord) for coord in cell["bbox"]]
        text = cell.get(text_key, "")

        if no_page_hf and cell.get("category") in ["Page-header", "Page-footer"]:
            continue

        if cell.get("category") == "Picture":
            image_crop = image.crop((x1, y1, x2, y2))
            image_data_url = pil_to_data_url_png(image_crop)
            text_items.append(f"![]({image_data_url})")
        elif cell.get("category") == "Formula":
            text_items.append(get_formula_in_markdown(text))
        else:
            text_items.append(clean_text(text))

    return "\n\n".join(text_items)


def fix_streamlit_formulas(md: str) -> str:
    def replace_formula(match):
        content = match.group(1)
        if content.startswith("\n"):
            content = content[1:]
        if content.endswith("\n"):
            content = content[:-1]
        return f"$$\n{content}\n$$"

    return re.sub(r"\$\$(.*?)\$\$", replace_formula, md, flags=re.DOTALL)


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def _extract_json_candidate(text: str) -> str:
    t = _strip_code_fences(text)
    # Try to capture the first JSON object/array blob.
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", t)
    return (m.group(1) if m else t).strip()


def extract_json_candidate(text: str) -> str:
    """Public wrapper used by the frontend for JSON/JSONL downloads."""
    return _extract_json_candidate(text)


def parse_json_from_response(response_text: str) -> Any:
    """Parse the first JSON object/array found in the response text."""
    candidate = _extract_json_candidate(response_text)
    return json.loads(candidate)


def parse_layout_cells_from_response(response_text: str) -> list[dict]:
    """
    The author prompt says "single JSON object", but in practice it may be:
    - a JSON list of cells
    - a JSON object containing a list under common keys
    - wrapped in ```json fences
    """
    candidate = _extract_json_candidate(response_text)
    obj: Any = json.loads(candidate)

    if isinstance(obj, list):
        return [c for c in obj if isinstance(c, dict)]

    if isinstance(obj, dict):
        for key in ("cells", "elements", "items", "layouts", "layout"):
            v = obj.get(key)
            if isinstance(v, list):
                return [c for c in v if isinstance(c, dict)]
        # Some models return {"bbox":..., ...} as a single cell.
        if "bbox" in obj and "category" in obj:
            return [obj]

    raise ValueError("Unsupported JSON shape for layout cells")


def response_to_markdown(*, prompt_mode: str, image: Image.Image, response_text: str) -> str | None:
    if prompt_mode not in {"prompt_layout_all_en"}:
        return None
    cells = parse_layout_cells_from_response(response_text)
    md = layoutjson2md(image, cells, text_key="text", no_page_hf=False)
    return fix_streamlit_formulas(md)

