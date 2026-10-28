from datetime import datetime
from pydantic import BaseModel, Field
class FileChange(BaseModel):
    path: str
    added: int
    removed: int
class Commit(BaseModel):
    sha: str
    author: str
    when: datetime
    subject: str
    files: list[FileChange] = Field(default_factory=list)
    @property
    def churn(self) -> int:
        return sum(f.added + f.removed for f in self.files)
class Session(BaseModel):
    start: datetime
    end: datetime
    commits: int
    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60
class BuildSignals(BaseModel):
    day: int
    title: str
    summary: str
    path: str
    files: int
    loc: int
    commits: int
    sessions: int
    active_days: int
    has_tests: bool
    has_demo: bool
    decisions: int
    failure_notes: int
    measured_results: int
class ScoredBuild(BaseModel):
    day: int
    title: str
    score: float
    role_fit: float
    parts: dict[str, float]
class Receipt(BaseModel):
    label: str
    value: float
    unit: str = ''
    source: str = 'git'
class Finding(BaseModel):
    raw: str
    value: float
    unit: str
    status: str
class CaseStudy(BaseModel):
    headline: str
    problem: str
    decision: str
    tradeoff: str
    result: str
    what_broke: str
class DraftResult(BaseModel):
    study: CaseStudy
    source: str
    attempts: int
    findings: list[Finding]
    error: str = ''
