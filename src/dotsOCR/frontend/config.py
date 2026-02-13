from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class UiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    title: str = Field(default="DotsOCR", validation_alias="FRONTEND_TITLE")
    vllm_base_url: str = Field(default="http://vllm:8000/v1", validation_alias="VLLM_BASE_URL")
    vllm_model: str = Field(default="dotsOCR", validation_alias="VLLM_MODEL_NAME")
    vllm_api_key: str = Field(default="0", validation_alias="VLLM_API_KEY")
    vllm_max_model_len: int = Field(default=12000, validation_alias="VLLM_MAX_MODEL_LEN")


def load_ui_config() -> UiConfig:
    # Cache per Streamlit process to avoid re-parsing env on reruns.
    return _cached_ui_config()


@lru_cache(maxsize=1)
def _cached_ui_config() -> UiConfig:
    return UiConfig()

