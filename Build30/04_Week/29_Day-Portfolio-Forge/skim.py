import re
from pydantic import BaseModel
from config import READ_BUDGET_SECONDS, WORDS_PER_MINUTE
BANNED = ('delve', 'foster', 'leverage', 'utilize', 'streamline', 'empower', 'robust', 'seamless', 'comprehensive', 'innovative', 'crucial')
PLACEHOLDER = re.compile(r'\[fill in[^\]]*\]', re.I)
REQUIRED = ('problem', 'decision', 'tradeoff', 'result')
CODE_LINE_SECONDS = 1.5
class SkimReport(BaseModel):
    seconds: float
    words: int
    code_lines: int
    issues: list[str]
    @property
    def passed(self) -> bool:
        return not self.issues
def split_code(markdown: str) -> tuple[str, int]:
    prose, code_lines, inside = [], 0, False
    for line in markdown.splitlines():
        if line.strip().startswith('```'):
            inside = not inside
        elif inside:
            code_lines += 1
        else:
            prose.append(line)
    return '\n'.join(prose), code_lines
def paragraphs(prose: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r'\n\s*\n', prose)]
    return [block for block in blocks if block and not block.startswith(('#', '-'))]
def skim(markdown: str, budget_seconds: int = READ_BUDGET_SECONDS, wpm: int = WORDS_PER_MINUTE) -> SkimReport:
    prose, code_lines = split_code(markdown)
    words = len(prose.split())
    seconds = round(words / wpm * 60 + code_lines * CODE_LINE_SECONDS, 1)
    issues = []
    if seconds > budget_seconds:
        issues.append(f'reads in {seconds} seconds, budget is {budget_seconds}')
    if code_lines > 15:
        issues.append(f'code block has {code_lines} lines, keep it at 15 or fewer')
    first_line = next((line for line in prose.splitlines() if line.startswith('# ')), '')
    if len(first_line.split()) - 1 > 12:
        issues.append('headline is longer than 12 words')
    body = paragraphs(prose)
    if body and not re.search(r'\d', ' '.join(' '.join(body).split()[:80])):
        issues.append('no number in the first 80 words')
    for block in body:
        if len(block.split()) > 70:
            issues.append('a paragraph is longer than 70 words')
            break
    lowered = prose.lower()
    for section in REQUIRED:
        if f'## {section}' not in lowered:
            issues.append(f'missing section: {section}')
    used = [word for word in BANNED if re.search(rf'\b{word}\b', lowered)]
    if used:
        issues.append('banned words: ' + ', '.join(used))
    holes = PLACEHOLDER.findall(prose)
    if holes:
        issues.append(f'{len(holes)} placeholders still to fill')
    return SkimReport(seconds=seconds, words=words, code_lines=code_lines, issues=issues)
