from schemas import CaseStudy, Receipt, ScoredBuild
from snippet import Snippet
DEMO_BANNER = 'DEMO DATA. This was generated from a practice repo. Do not publish it.'
SYMBOL_UNITS = {'%', 'h', 'ms', 's', 'x', 'min'}
def receipt_text(receipt: Receipt) -> str:
    unit = receipt.unit if receipt.unit in SYMBOL_UNITS else ''
    return f'- {receipt.label}: {receipt.value:g}{unit} ({receipt.source})'
def render_case_study(study: CaseStudy, author: str, day: int, receipts: list[Receipt], snippet: Snippet | None, demo: bool = False) -> str:
    lines = []
    if demo:
        lines += [DEMO_BANNER, '']
    lines += [f'# {study.headline}', '', f'Written by {author}. Build30 day {day}.', '']
    for title, text in (('Problem', study.problem), ('Decision', study.decision), ('Tradeoff', study.tradeoff), ('Result', study.result), ('What broke and how I fixed it', study.what_broke)):
        lines += [f'## {title}', '', text, '']
    if snippet:
        lines += ['## Code that matters', '', f'{snippet.name} in {snippet.path}', '', '```python', snippet.code, '```', '']
    lines += ['## Receipts', '']
    lines += [receipt_text(receipt) for receipt in receipts]
    return '\n'.join(lines) + '\n'
def render_portfolio(author: str, role: str, ranked: list[ScoredBuild], lead: ScoredBuild, stats: dict, demo: bool = False) -> str:
    lines = []
    if demo:
        lines += [DEMO_BANNER, '']
    lines += [f'# {author}, portfolio', '', f'Target role: {role}.', '', 'Start here: CASE_STUDY.md', '']
    lines += ['## Build history', '']
    lines += [f'- {label}: {value}' for label, value in stats.items()]
    lines += ['', '## Why this build leads', '']
    top = sorted(lead.parts.items(), key=lambda item: item[1], reverse=True)[:3]
    lines += [f'- {name}: {value:.2f}' for name, value in top]
    lines += ['', '## All builds by score', '']
    lines += [f'- Day {item.day}, {item.title}: {item.score:.2f}' for item in ranked]
    return '\n'.join(lines) + '\n'
