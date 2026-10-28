from typing import Literal, Optional
from pydantic import BaseModel, Field
class Brief(BaseModel):
    client: str
    stated_problem: str = Field(description='What the client asked for, in their own words')
    underlying_problem: str = Field(description='The business pain behind the ask')
    current_process: str = Field(description='How the work is done today, step by step')
    volume_per_month: Optional[int] = None
    minutes_per_task: Optional[float] = None
    cost_per_hour_usd: Optional[float] = None
    automatable_share: Optional[float] = Field(default=None, ge=0, le=1, description='Share of the volume the build can handle, from 0 to 1')
    baseline_metric: Optional[str] = Field(default=None, description='A measured number that describes today')
    success_metric: Optional[str] = Field(default=None, description='The number the client will sign off on')
    data_sources: list[str] = []
    data_access: Literal['ready', 'needs_work', 'none', 'unknown'] = 'unknown'
    owner: Optional[str] = Field(default=None, description='One named person on the client side who can say yes or no')
    budget_usd: Optional[int] = None
    risks: list[str] = []
    open_questions: list[str] = []
