from .fixed import FixedChunker
from .sentence import SentenceChunker
from .recursive import RecursiveChunker
STRATEGIES = {
    'fixed': FixedChunker,
    'sentence': SentenceChunker,
    'recursive': RecursiveChunker,
}
def get_chunker(name, counter, size=200, overlap=40):
    if name not in STRATEGIES:
        raise ValueError(f'unknown strategy {name}, pick one of {list(STRATEGIES)}')
    return STRATEGIES[name](counter, size=size, overlap=overlap)
