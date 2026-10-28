import re
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
CARD_RE = re.compile(r'(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)')
PHONE_RE = re.compile(r'(?<![\w])\+?\d[\d ()-]{8,16}\d(?![\w])')
def luhn_ok(number):
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
def mask_pii_entities(text):
    counts = {'EMAIL': 0, 'CARD': 0, 'PHONE': 0}
    def card(match):
        if luhn_ok(match.group()):
            counts['CARD'] += 1
            return '[CARD]'
        return match.group()
    def email(match):
        counts['EMAIL'] += 1
        return '[EMAIL]'
    def phone(match):
        digits = sum(c.isdigit() for c in match.group())
        if 10 <= digits <= 13:
            counts['PHONE'] += 1
            return '[PHONE]'
        return match.group()
    text = CARD_RE.sub(card, text)
    text = EMAIL_RE.sub(email, text)
    text = PHONE_RE.sub(phone, text)
    return text, counts
