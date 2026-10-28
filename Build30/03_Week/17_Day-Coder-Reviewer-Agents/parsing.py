import re
FENCE = chr(96) * 3
CODE_BLOCK = re.compile(FENCE + r'(?:python|py)?[ \t]*\n(.*?)' + FENCE, re.DOTALL)
VERDICT_LINE = re.compile(r'^VERDICT:\s*(APPROVED|REVISE)\s*$', re.IGNORECASE)
def extract_code(text):
    match = CODE_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()
def parse_verdict(text):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return 'REVISE', 'Reviewer returned nothing.'
    first = lines[0].strip('*#> ').strip()
    match = VERDICT_LINE.match(first)
    if not match:
        return 'REVISE', text.strip()
    verdict = match.group(1).upper()
    feedback = '\n'.join(lines[1:])
    return verdict, feedback
