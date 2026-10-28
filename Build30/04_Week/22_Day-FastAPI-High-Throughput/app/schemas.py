from pydantic import BaseModel, Field
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)
class SourceChunk(BaseModel):
    id: int
    text: str
    score: float
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
class AskResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[SourceChunk]
    usage: Usage
    latency_ms: float
