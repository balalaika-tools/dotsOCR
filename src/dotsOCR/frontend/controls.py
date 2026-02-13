from __future__ import annotations

import streamlit as st


def sidebar_controls(*, running: bool) -> dict:
    st.sidebar.header("OCR")

    # Exactly 2 modes, fixed labels.
    label_to_mode = {
        "Layout+Text": "prompt_layout_all_en",
        "Text": "prompt_ocr",
    }
    selected_label = st.sidebar.radio("Mode", options=["Layout+Text", "Text"], index=0, disabled=running)
    selected_mode = label_to_mode[selected_label]

    dpi_presets = [100, 150, 200, 240, 300]
    target_dpi = st.sidebar.select_slider("PDF render DPI", options=dpi_presets, value=200, disabled=running)
    max_pages = st.sidebar.number_input("Max pages (0 = all)", min_value=0, max_value=500, value=0, step=1, disabled=running)
    max_in_flight = st.sidebar.slider("Concurrency (requests in flight)", min_value=1, max_value=100, value=50, disabled=running)

    return {
        "selected_mode": selected_mode,
        "target_dpi": int(target_dpi),
        "max_pages": None if int(max_pages) == 0 else int(max_pages),
        "max_in_flight": int(max_in_flight),
    }

