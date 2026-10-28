import re
import secrets
def make_canary():
    return 'CANARY-' + secrets.token_hex(4)
def shingles(text, size=6):
    words = re.findall(r'\w+', text.lower())
    return {' '.join(words[i:i + size]) for i in range(max(len(words) - size + 1, 0))}
def check_output(answer, system_prompt, canary):
    reasons = []
    if canary.lower() in answer.lower():
        reasons.append('canary_leaked')
    if shingles(system_prompt) & shingles(answer):
        reasons.append('system_prompt_overlap')
    return {'leaked': bool(reasons), 'reasons': reasons}
