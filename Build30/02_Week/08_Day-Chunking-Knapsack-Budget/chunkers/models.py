from dataclasses import dataclass
@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int
    start_char: int
    end_char: int
    token_count: int
    strategy: str
    heading: str = ''
    @property
    def chunk_id(self):
        return f'{self.source}::{self.index}'
