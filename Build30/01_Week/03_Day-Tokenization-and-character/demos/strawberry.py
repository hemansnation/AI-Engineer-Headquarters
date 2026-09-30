from demos.tiktoken_lab import ENCODINGS, load, pieces
from tracker import log_event


def run(word='strawberry', letter='r'):
    print(f'Python counts {letter} in {word}: {word.count(letter)}')
    for name in ENCODINGS:
        enc = load(name)
        if enc is None:
            continue
        ids, parts = pieces(enc, word)
        print(f'{name}: ids={ids} pieces={parts}')
        log_event('strawberry', word, {name: len(ids)})
    print('The model receives the ids. It never receives the letters.')


def ask_gemini(word='strawberry', letter='r'):
    from gemini_client import ask
    prompt = f'How many times does the letter {letter} appear in {word}? Reply with one number only.'
    result = ask(prompt)
    answer = result['text'].strip()
    print(f'Gemini says: {answer} (real answer: {word.count(letter)})')
    print('prompt tokens:', result['prompt_tokens'])
    print('output tokens:', result['output_tokens'])
    print('thinking tokens:', result['thinking_tokens'])
    print('total tokens:', result['total_tokens'])
    log_event('gemini_letter_count', prompt, {'gemini_total': result['total_tokens']})