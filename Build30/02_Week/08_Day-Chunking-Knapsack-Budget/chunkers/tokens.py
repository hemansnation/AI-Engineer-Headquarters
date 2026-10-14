import re
class WordCounter:
    name = 'words'
    def offsets(self, text):
        return [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
    def count(self, text):
        return len(self.offsets(text))
class TokenCounter:
    def __init__(self, model_name='BAAI/bge-small-en-v1.5'):
        from transformers import AutoTokenizer
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name, model_max_length=1_000_000)
    def offsets(self, text):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(s, e) for s, e in enc['offset_mapping']]
    def count(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))
def get_counter(name='BAAI/bge-small-en-v1.5'):
    if name == 'words':
        return WordCounter()
    return TokenCounter(name)
