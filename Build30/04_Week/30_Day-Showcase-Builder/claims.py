import re
NUMBER = re.compile(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?')
PLACEHOLDER = '[NUMBER NEEDED]'
def normalize(raw):
    cleaned = raw.replace(',', '')
    whole, _, fraction = cleaned.partition('.')
    whole = whole.lstrip('0') or '0'
    fraction = fraction.rstrip('0')
    return whole + ('.' + fraction if fraction else '')
def numbers_in(text):
    return {normalize(raw) for raw in NUMBER.findall(text)}
def allowed_numbers(sources):
    allowed = set()
    for text in sources:
        allowed |= numbers_in(text)
    return allowed
def unsupported(draft, sources):
    return sorted(numbers_in(draft) - allowed_numbers(sources))
def mask(draft, bad):
    def swap(match):
        return PLACEHOLDER if normalize(match.group()) in bad else match.group()
    return NUMBER.sub(swap, draft)
