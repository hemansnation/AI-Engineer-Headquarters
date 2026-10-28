from datetime import date, timedelta
from schemas import Commit, Session
def make_sessions(commits: list[Commit], gap_minutes: int = 90) -> list[Session]:
    gap = timedelta(minutes=gap_minutes)
    sessions: list[Session] = []
    for commit in sorted(commits, key=lambda item: item.when):
        if sessions and commit.when - sessions[-1].end <= gap:
            sessions[-1].end = commit.when
            sessions[-1].commits += 1
        else:
            sessions.append(Session(start=commit.when, end=commit.when, commits=1))
    return sessions
def active_dates(commits: list[Commit]) -> list[date]:
    return sorted({commit.when.date() for commit in commits})
def longest_streak(commits: list[Commit]) -> int:
    best = run = 0
    previous = None
    for day in active_dates(commits):
        run = run + 1 if previous and day - previous == timedelta(days=1) else 1
        best = max(best, run)
        previous = day
    return best
