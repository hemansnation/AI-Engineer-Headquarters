import os
import re
from pathlib import Path
from config import SESSION_GAP_MINUTES
from gitlog import day_of
from schemas import BuildSignals, Commit
from sessions import active_dates, make_sessions
SKIP_DIRS = {'.git', 'venv', '.venv', 'node_modules', '__pycache__', 'out'}
CODE_SUFFIXES = {'.py', '.js', '.ts', '.sh'}
DEMO_SUFFIXES = {'.gif', '.mp4', '.mov', '.png', '.webm'}
DAY_DIR = re.compile(r'^day[-_ ]?0*(\d{1,2})(?:\D.*)?$', re.I)
DECISION = re.compile(r'\b(because|instead of|trade-?off|chose|decided|rejected|switched)\b', re.I)
FAILURE = re.compile(r'\b(broke|failed|failure|bug|limitation|still fails|wrong)\b', re.I)
RESULT = re.compile(r'\d+(?:[.,]\d+)?\s?(?:%|ms|s|x|k|tokens|seconds|minutes|queries|users)(?![a-z])', re.I)
def keep_dirs(root: str, dirs: list[str]) -> list[str]:
    return [name for name in dirs if name not in SKIP_DIRS and not (Path(root) / name / '.git').exists()]
def find_day_dirs(repo: Path) -> dict[int, Path]:
    found = {}
    for root, dirs, _ in os.walk(repo):
        dirs[:] = keep_dirs(root, dirs)
        for name in dirs:
            match = DAY_DIR.match(name)
            if match:
                found[int(match.group(1))] = Path(root) / name
    return found
def walk_files(folder: Path) -> list[Path]:
    files = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = keep_dirs(root, dirs)
        files.extend(Path(root) / name for name in names)
    return files
def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')
def read_readme(folder: Path) -> str:
    for name in ('README.md', 'readme.md'):
        if (folder / name).exists():
            return read_text(folder / name)
    return ''
def clean_title(title: str) -> str:
    return re.sub(r'^day[-_ ]?\d+\s*[:.\-]?\s*', '', title, flags=re.I).strip() or title
def title_of(folder: Path, readme: str) -> str:
    for line in readme.splitlines():
        if line.startswith('#'):
            return clean_title(line.lstrip('#').strip())
    return folder.name
def summary_of(readme: str, limit: int = 600) -> str:
    body = [line for line in readme.splitlines() if line.strip() and not line.startswith('#')]
    return ' '.join(body)[:limit]
def scan_day(day: int, folder: Path, commits: list[Commit]) -> BuildSignals:
    readme = read_readme(folder)
    files = walk_files(folder)
    code = [path for path in files if path.suffix in CODE_SUFFIXES]
    return BuildSignals(
        day=day,
        title=title_of(folder, readme),
        summary=summary_of(readme),
        path=str(folder),
        files=len(files),
        loc=sum(len(read_text(path).splitlines()) for path in code),
        commits=len(commits),
        sessions=len(make_sessions(commits, SESSION_GAP_MINUTES)),
        active_days=len(active_dates(commits)),
        has_tests=any(path.name.startswith('test_') or 'tests' in path.parts for path in files),
        has_demo=any(path.suffix.lower() in DEMO_SUFFIXES for path in files),
        decisions=len(DECISION.findall(readme)),
        failure_notes=len(FAILURE.findall(readme)),
        measured_results=len(RESULT.findall(readme)),
    )
def scan_all(repo: Path, grouped: dict[int, list[Commit]]) -> list[BuildSignals]:
    builds = [scan_day(day, folder, grouped.get(day, [])) for day, folder in find_day_dirs(repo).items()]
    return sorted(builds, key=lambda build: build.day)
