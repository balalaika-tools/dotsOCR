# DotsOCR (vLLM + Streamlit)

Run **DotsOCR** as an OpenAI-compatible **vLLM** server with a **Streamlit** UI for OCR and layout extraction on images and multi-page PDFs.

## Requirements

- **Docker** and **Docker Compose**
- **NVIDIA GPU** with enough VRAM to load the model (vLLM runs on GPU only)

This setup was developed and tested on an **NVIDIA RTX 2080 Ti**. If you use a different GPU, you may need to adjust `VLLM_GPU_MEMORY_UTILIZATION` or `VLLM_MAX_MODEL_LEN` in `.env`.

## Quick start

1. Copy `.env.example` to `.env` and set at least:
   - `HF_TOKEN` — your [Hugging Face token](https://huggingface.co/settings/tokens) (for model download)
   - Optionally `HF_CACHE_HOST_PATH` — host path for the Hugging Face cache (e.g. `C:/Users/you/hf_cache` on Windows)

2. Start services:

```bash
docker compose --env-file .env up -d --build
```

3. Open the UI at **http://localhost:8501** (or the port set by `FRONTEND_PORT` in `.env`).

## Services

| Service     | URL / port | Description                    |
|------------|------------|--------------------------------|
| **vLLM**   | `http://localhost:8000/v1` (default) | OpenAI-compatible API for the Dots OCR model |
| **Streamlit** | `http://localhost:8501` (default) | Web UI: upload images/PDFs, run inference, download results |

Ports can be changed via `VLLM_PORT` and `FRONTEND_PORT` in `.env`.

## Configuration

See `.env.example` for all options. Important ones:

- **vLLM**: `VLLM_MODEL`, `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_TENSOR_PARALLEL_SIZE`
- **Streamlit**: `FRONTEND_PORT`, `FRONTEND_TITLE`
- **Hugging Face**: `HF_TOKEN`, `HF_CACHE_HOST_PATH` (for cache bind-mount)

