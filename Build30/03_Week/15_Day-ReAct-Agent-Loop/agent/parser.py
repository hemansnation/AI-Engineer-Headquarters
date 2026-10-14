import re
THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL)
FINAL = re.compile(r'Final Answer:\s*(.*)', re.DOTALL)
ACTION = re.compile(r'Action:\s*(\w+)\s*\n\s*Action Input:\s*(.*)', re.DOTALL)
QUOTES = chr(34) + chr(39) + chr(96)
def clean(text):
    return THINK_BLOCK.sub('', text).strip()
def thought_before(text, position):
    return text[:position].replace('Thought:', '').strip()
def parse(text):
    text = clean(text)
    final = FINAL.search(text)
    action = ACTION.search(text)
    if action and (not final or action.start() < final.start()):
        first_line = action.group(2).strip().split('\n')[0]
        return {
            'kind': 'action',
            'tool': action.group(1).strip(),
            'input': first_line.strip().strip(QUOTES).strip(),
            'thought': thought_before(text, action.start()),
        }
    if final:
        return {
            'kind': 'final',
            'answer': final.group(1).strip(),
            'thought': thought_before(text, final.start()),
        }
    return {'kind': 'invalid', 'raw': text, 'thought': ''}
