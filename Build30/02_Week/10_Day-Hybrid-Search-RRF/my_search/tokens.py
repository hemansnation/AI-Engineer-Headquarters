import re
TOKEN_PATTERN = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)*')
def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())
