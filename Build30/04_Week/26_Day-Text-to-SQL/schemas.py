from typing import Optional
from pydantic import BaseModel, Field
class SQLDraft(BaseModel):
    sql: str = Field(description='One SQLite SELECT statement that answers the question')
    tables: list[str] = Field(description='Tables the query reads')
    assumption: Optional[str] = Field(default=None, description='Anything you assumed because the question was unclear')
