import heapq
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from config import DAY_PATTERN
from schemas import Commit, FileChange
RECORD = '\x1e'
FIELD = '\x1f'
FORMAT = '%x1e%H%x1f%an%x1f%aI%x1f%s'
DECISION_WORDS = re.compile(r'\b(because|instead of|trade-?off|chose|decided|rejected|switched?|replaced?|reverted?|fix(?:ed)?|broke)\b', re.I)
def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True)
    return result.stdout
def parse_numstat(lines: list[str]) -> list[FileChange]:
    changes = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) != 3:
            continue
        added, removed, path = parts
        changes.append(FileChange(
            path=path,
            added=int(added) if added.isdigit() else 0,
            removed=int(removed) if removed.isdigit() else 0,
        ))
    return changes
def read_commits(repo: Path) -> list[Commit]:
    raw = run_git(repo, 'log', '--numstat', '--no-merges', '--pretty=format:' + FORMAT)
    commits = []
    for block in raw.split(RECORD):
        lines = block.strip('\n').split('\n')
        if not lines or not lines[0].strip():
            continue
        sha, author, when, subject = lines[0].split(FIELD, 3)
        commits.append(Commit(
            sha=sha,
            author=author,
            when=datetime.fromisoformat(when),
            subject=subject,
            files=parse_numstat(lines[1:]),
        ))
    commits.sort(key=lambda commit: commit.when)
    return commits
def day_of(path: str) -> int | None:
    match = re.search(DAY_PATTERN, path)
    return int(match.group(1)) if match else None
def commits_by_day(commits: list[Commit]) -> dict[int, list[Commit]]:
    grouped = defaultdict(list)
    for commit in commits:
        days = {day_of(change.path) for change in commit.files}
        days.discard(None)
        for day in days:
            grouped[day].append(commit)
    return dict(grouped)
def day_churn(commits: list[Commit], day: int) -> tuple[int, int]:
    added = removed = 0
    for commit in commits:
        for change in commit.files:
            if day_of(change.path) == day:
                added += change.added
                removed += change.removed
    return added, removed
def decision_subjects(commits: list[Commit], day: int, k: int = 3) -> list[str]:
    scored = []
    for commit in commits:
        if DECISION_WORDS.search(commit.subject):
            changed = sum(c.added + c.removed for c in commit.files if day_of(c.path) == day)
            scored.append((changed, commit.subject))
    return [subject for _, subject in heapq.nlargest(k, scored)]
