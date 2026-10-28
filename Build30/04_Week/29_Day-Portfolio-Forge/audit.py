import re
import sys
from pathlib import Path
from receipts import failures, verify
from schemas import Receipt
from skim import skim, split_code
LINE = re.compile(r'^- (.+): ([\d.]+)(%|h|ms|s|x|min)? \((.+)\)$')
def split_receipts(markdown: str) -> tuple[str, list[Receipt]]:
    body, _, tail = markdown.partition('## Receipts')
    receipts = []
    for line in tail.splitlines():
        match = LINE.match(line.strip())
        if match:
            label, value, unit, source = match.groups()
            receipts.append(Receipt(label=label, value=float(value), unit=unit or '', source=source))
    return split_code(body)[0], receipts
def audit(markdown: str) -> list[str]:
    body, receipts = split_receipts(markdown)
    problems = [f'{item.raw} is {item.status}' for item in failures(verify(body, receipts))]
    return problems + skim(markdown).issues
if __name__ == '__main__':
    found = audit(Path(sys.argv[1]).read_text(encoding='utf-8'))
    print('\n'.join(found) if found else 'clean')
    sys.exit(1 if found else 0)
