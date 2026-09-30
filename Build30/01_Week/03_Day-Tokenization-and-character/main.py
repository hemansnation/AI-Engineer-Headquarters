import argparse
from pathlib import Path

from mytok.bpe_tokenizer import BPETokenizer
from mytok.char_tokenizer import CharTokenizer
from mytok.word_tokenizer import WordTokenizer
from tracker import log_event

CORPUS = Path('corpus.txt').read_text(encoding='utf-8')


def lab_chars():
    tok = CharTokenizer()
    tok.train(CORPUS)
    text = 'tokens cost money'
    ids = tok.encode(text)
    print('vocab size:', len(tok.stoi))
    print('ids:', ids)
    print('round trip ok:', tok.decode(ids) == text)
    print('tokens for text:', len(ids), 'chars:', len(text))
    log_event('char_tokenizer', text, {'tokens': len(ids)})
    try:
        tok.encode('hello नमस्ते')
    except KeyError as err:
        print('unknown character breaks it:', err)


def lab_words():
    tok = WordTokenizer()
    tok.train(CORPUS)
    text = 'Tokens cost money, so count tokens before strawberry season.'
    ids = tok.encode(text)
    print('vocab size:', len(tok.stoi))
    print('ids:', ids)
    print('decoded:', tok.decode(ids))
    print('round trip exact:', tok.decode(ids) == text)
    log_event('word_tokenizer', text, {'tokens': len(ids)})


def lab_bpe():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=320)
    print('merges learned:', len(tok.merges))
    for (a, b), new_id in list(tok.merges.items())[:10]:
        print(f'  {tok.vocab[a]} + {tok.vocab[b]} -> {tok.vocab[new_id]} (id {new_id})')
    text = 'the tokenizer counts tokens'
    ids = tok.encode(text)
    raw_len = len(text.encode('utf-8'))
    print('pieces:', tok.pieces(ids))
    print(f'bytes={raw_len} tokens={len(ids)} compression={raw_len / len(ids):.2f}x')
    print('round trip ok:', tok.decode(ids) == text)
    unseen = 'नमस्ते'
    unseen_ids = tok.encode(unseen)
    print('unseen script still works:', tok.decode(unseen_ids) == unseen, 'tokens:', len(unseen_ids))
    log_event('bpe', text, {'tokens': len(ids)})


def lab_strawberry(with_gemini):
    from demos.strawberry import ask_gemini, run
    run()
    if with_gemini:
        ask_gemini()


def lab_cost(with_gemini):
    from demos.multilingual_cost import run_gemini, run_own_bpe, run_tiktoken
    run_own_bpe(CORPUS)
    run_tiktoken()
    if with_gemini:
        run_gemini()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lab', choices=['chars', 'words', 'bpe', 'strawberry', 'cost', 'all'])
    parser.add_argument('--gemini', action='store_true')
    args = parser.parse_args()
    labs = {
        'chars': lab_chars,
        'words': lab_words,
        'bpe': lab_bpe,
        'strawberry': lambda: lab_strawberry(args.gemini),
        'cost': lambda: lab_cost(args.gemini),
    }
    if args.lab == 'all':
        for name, fn in labs.items():
            print(f'\n=== {name} ===')
            fn()
    else:
        labs[args.lab]()


if __name__ == '__main__':
    main()