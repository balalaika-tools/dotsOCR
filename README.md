# DotsOCR (vLLM + Streamlit)

This repo runs **DotsOCR** as an OpenAI-compatible **vLLM** server and a **Streamlit** UI for testing OCR / layout prompts on images and multi-page PDFs.

## Quick start

1. Copy `.env.example` to `.env` and set `HF_TOKEN` (and optionally `HF_CACHE_HOST_PATH`).
2. Start everything:

```bash
docker compose --env-file .env up -d --build
```

## Services

- **vLLM**: `http://localhost:${VLLM_PORT:-8000}/v1`
- **Streamlit UI**: `http://localhost:${FRONTEND_PORT:-8501}`

