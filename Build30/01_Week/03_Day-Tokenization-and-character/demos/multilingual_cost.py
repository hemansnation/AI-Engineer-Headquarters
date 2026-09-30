from demos.tiktoken_lab import ENCODINGS, load
from mytok.bpe_tokenizer import BPETokenizer
from tracker import log_event

SENTENCES = {
    'English': 'Good morning, how are you today?',
    'Hindi': 'सुप्रभात, आप आज कैसे हैं?',
    'Japanese': 'おはようございます、今日はお元気ですか?',
}

PRICE_PER_MILLION_TOKENS = 1.00


def cost_of(tokens):
    return tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS


def run_own_bpe(corpus_text):
    bpe = BPETokenizer()
    bpe.train(corpus_text, vocab_size=320)
    print('Our own BPE, trained on English only:')
    for lang, text in SENTENCES.items():
        ids = bpe.encode(text)
        raw_bytes = len(text.encode('utf-8'))
        print(f'  {lang}: chars={len(text)} bytes={raw_bytes} tokens={len(ids)}')
        log_event('own_bpe', text, {'tokens': len(ids)})


def run_tiktoken():
    for name in ENCODINGS:
        enc = load(name)
        if enc is None:
            continue
        print(f'{name}:')
        english_tokens = None
        for lang, text in SENTENCES.items():
            tokens = len(enc.encode(text))
            if english_tokens is None:
                english_tokens = tokens
            ratio = tokens / english_tokens
            print(f'  {lang}: tokens={tokens} tokens_per_char={tokens / len(text):.2f} vs English={ratio:.2f}x cost={cost_of(tokens):.8f}')
            log_event('tiktoken_' + name, text, {'tokens': tokens})


def run_gemini():
    from gemini_client import count_tokens
    print('Gemini count_tokens (free call):')
    for lang, text in SENTENCES.items():
        tokens = count_tokens(text)
        print(f'  {lang}: tokens={tokens}')
        log_event('gemini_count', text, {'tokens': tokens})