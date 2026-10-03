from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Settings:
    name: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    max_output_tokens: int = 200
    stop_sequences: tuple = ()

    def tweak(self, **changes):
        return replace(self, **changes)

    def to_dict(self):
        raw = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'seed': self.seed,
            'max_output_tokens': self.max_output_tokens,
            'stop_sequences': list(self.stop_sequences) or None,
        }
        return {key: value for key, value in raw.items() if value is not None}

PRECISE = Settings(name='precise', temperature=0.0)
BALANCED = Settings(name='balanced', temperature=0.7, top_p=0.95)
WILD = Settings(name='wild', temperature=1.5)
SEEDED = BALANCED.tweak(name='seeded', seed=42)

PRESETS = {p.name: p for p in (PRECISE, BALANCED, WILD, SEEDED)}