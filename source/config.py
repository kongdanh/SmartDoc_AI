import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "openai"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3"
    llm_api_key: str = "EMPTY"
    llm_rpm: int = 1000

    embedding_provider: str = "openai"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "nomic-embed-text"
    embedding_api_key: str = "EMPTY"
    embedding_rpm: int = 1000
    embedding_dimension: int = 1536

    indexing_method: str = "fast"

    data_dir: str = "./data"
    index_dir: str = "./indexes"
    server_port: int = 8001
    use_proxy: bool = True
    http_proxy: str = ""
    https_proxy: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir).resolve()

    @property
    def settings_template_path(self) -> Path:
        return Path(__file__).parent.parent / "settings_template.yaml"


settings = Settings()

if settings.use_proxy:
    os.environ.setdefault("HTTP_PROXY", settings.http_proxy)
    os.environ.setdefault("HTTPS_PROXY", settings.https_proxy)
else:
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)

_PROVIDER_ENV_KEYS = {
    "openrouter": "OPENROUTER_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}
for _provider, _key in [
    (settings.llm_provider, settings.llm_api_key),
    (settings.embedding_provider, settings.embedding_api_key),
]:
    env_name = _PROVIDER_ENV_KEYS.get(_provider)
    if env_name and _key != "EMPTY":
        os.environ.setdefault(env_name, _key)

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
