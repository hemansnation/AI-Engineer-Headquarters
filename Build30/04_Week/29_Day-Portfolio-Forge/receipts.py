import re
import tomllib
from pathlib import Path
from config import SESSION_GAP_MINUTES
from gitlog import day_churn
from schemas import BuildSignals, Commit, Finding, Receipt
from sessions import make_sessions
UNITS = r'%|ms|s|x|k|h|hrs?|hours?|mins?|minutes?|secs?|seconds?|days?|commits?|sessions?|files?|lines?|tests?|tokens?|builds?'
NUMBER = re.compile(r'(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s?(' + UNITS + r')(?![a-z])|(?![a-z\d]))', re.I)
UNIT_MAP = {
    'hr': 'h', 'hrs': 'h', 'hour': 'h', 'hours': 'h', 'h': 'h',
    'min': 'min', 'mins': 'min', 'minute': 'min', 'minutes': 'min',
    'sec': 's', 'secs': 's', 'second': 's', 'seconds': 's', 's': 's',
    'day': 'day', 'days': 'day', 'commit': 'commit', 'commits': 'commit',
    'session': 'session', 'sessions': 'session', 'file': 'file', 'files': 'file',
    'line': 'line', 'lines': 'line', 'test': 'test', 'tests': 'test',
    'token': 'token', 'tokens': 'token', 'build': 'build', 'builds': 'build',
}
BAD = {'unsupported', 'unit_mismatch'}
def canon(unit: str) -> str:
    unit = unit.lower().strip()
    return UNIT_MAP.get(unit, unit)
def extract_numbers(text: str) -> list[tuple[str, float, int, str]]:
    found = []
    for match in NUMBER.finditer(text):
        whole, decimals, unit = match.group(1), match.group(2) or '', match.group(3) or ''
        value = float(whole.replace(',', '') + decimals)
        places = len(decimals) - 1 if decimals else 0
        unit = canon(unit)
        if unit == 'k':
            value, unit = value * 1000, ''
        if not unit and places == 0 and 1900 <= value <= 2100:
            continue
        found.append((match.group(0).strip(), value, places, unit))
    return found
def check_one(value: float, places: int, unit: str, receipts: list[Receipt]) -> str:
    same_value = [r for r in receipts if round(r.value, places) == round(value, places)]
    if not same_value:
        return 'unsupported'
    if unit and not any(canon(r.unit) in ('', unit) for r in same_value):
        return 'unit_mismatch'
    exact = any(r.value == value for r in same_value)
    return 'supported' if exact else 'rounded'
def verify(text: str, receipts: list[Receipt]) -> list[Finding]:
    findings = []
    for raw, value, places, unit in extract_numbers(text):
        findings.append(Finding(raw=raw, value=value, unit=unit, status=check_one(value, places, unit, receipts)))
    return findings
def failures(findings: list[Finding]) -> list[Finding]:
    return [finding for finding in findings if finding.status in BAD]
def toml_receipts(folder: Path) -> list[Receipt]:
    path = folder / 'receipts.toml'
    if not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding='utf-8'))
    return [
        Receipt(label=item['label'], value=float(item['value']), unit=item.get('unit', ''), source='receipts.toml')
        for item in data.get('receipt', [])
    ]
def build_receipts(build: BuildSignals, commits: list[Commit]) -> list[Receipt]:
    sessions = make_sessions(commits, SESSION_GAP_MINUTES)
    hours = round(sum(session.minutes for session in sessions) / 60, 1)
    added, removed = day_churn(commits, build.day)
    receipts = [
        Receipt(label='build day', value=build.day),
        Receipt(label='commits', value=build.commits, unit='commit'),
        Receipt(label='work sessions', value=build.sessions, unit='session'),
        Receipt(label='active days', value=build.active_days, unit='day'),
        Receipt(label='hours inside sessions', value=hours, unit='h'),
        Receipt(label='files', value=build.files, unit='file'),
        Receipt(label='lines of code', value=build.loc, unit='line'),
        Receipt(label='lines added', value=added, unit='line'),
        Receipt(label='lines removed', value=removed, unit='line'),
    ]
    return receipts + toml_receipts(Path(build.path))
