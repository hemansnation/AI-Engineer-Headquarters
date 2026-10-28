import re
import unicodedata
from guard.ahocorasick import AhoCorasick
ZERO_WIDTH = dict.fromkeys(map(ord, '​‌‍⁠﻿'))
INJECTION_PHRASES = [
    'ignore previous instructions',
    'ignore all previous instructions',
    'ignore the above',
    'disregard previous instructions',
    'disregard your instructions',
    'forget your instructions',
    'forget everything above',
    'you are now',
    'new instructions:',
    'show your system prompt',
    'reveal your system prompt',
    'print your system prompt',
    'repeat your system prompt',
    'reveal your instructions',
    'developer mode',
    'do anything now',
    'bypass your safety',
    'override your rules',
    'ai assistant:',
    'do not tell the user',
    'forward this email to',
    'forward the last',
    'send all customer',
]
def normalize(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(ZERO_WIDTH)
    text = text.lower()
    return re.sub(r'\s+', ' ', text).strip()
_automaton = AhoCorasick(INJECTION_PHRASES)
def scan_injection(text):
    clean = normalize(text)
    found = sorted({pattern for _, pattern in _automaton.search(clean)})
    return {'hit': bool(found), 'matches': found}
