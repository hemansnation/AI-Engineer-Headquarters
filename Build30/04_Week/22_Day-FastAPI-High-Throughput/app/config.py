from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    llm_mode: str = 'mock'
    ollama_url: str = 'http://localhost:11434'
    ollama_model: str = 'gemma4:e2b'
    embed_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    top_k: int = 3
    max_concurrent_llm_calls: int = 4
    queue_wait_seconds: float = 5.0
    llm_timeout_seconds: float = 60.0
    rate_limit_capacity: int = 10
    rate_limit_refill_per_second: float = 2.0
    usage_log_path: str = 'logs/usage.jsonl'
    corpus_path: str = 'data/corpus.txt'
@lru_cache
def get_settings() -> Settings:
    return Settings()
