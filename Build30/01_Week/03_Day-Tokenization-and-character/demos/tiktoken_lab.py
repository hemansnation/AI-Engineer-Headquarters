import tiktoken

ENCODINGS = ['cl100k_base', 'o200k_base']


def load(name):
    try:
        return tiktoken.get_encoding(name)
    except Exception as err:
        print(f'Could not load {name}. Check your internet once, tiktoken downloads it the first time.')
        print(err)
        return None


def pieces(enc, text):
    ids = enc.encode(text)
    parts = [enc.decode_single_token_bytes(i).decode('utf-8', errors='replace') for i in ids]
    return ids, parts