from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    app_env: str = 'local'
    log_level: str = 'INFO'
    embed_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    docs_path: str = 'data/docs.txt'
    top_k: int = 3
    cache_size: int = 256
    llm_base_url: str = ''
    llm_model: str = 'qwen2.5:1.5b'
    llm_api_key: str = 'not-needed'
    llm_timeout: float = 30.0


settings = Settings()
