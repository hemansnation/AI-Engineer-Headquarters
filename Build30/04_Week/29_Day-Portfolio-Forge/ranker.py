import heapq
from schemas import BuildSignals, ScoredBuild
WEIGHTS = {
    'decisions': 0.20,
    'results': 0.20,
    'failures': 0.15,
    'role_fit': 0.15,
    'demo': 0.10,
    'tests': 0.10,
    'depth': 0.10,
}
def saturate(value: float, cap: float) -> float:
    return min(value, cap) / cap
def rescale(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]
def parts_for(build: BuildSignals, fit: float) -> dict[str, float]:
    return {
        'decisions': saturate(build.decisions, 4),
        'results': saturate(build.measured_results, 3),
        'failures': saturate(build.failure_notes, 2),
        'role_fit': fit,
        'demo': 1.0 if build.has_demo else 0.0,
        'tests': 1.0 if build.has_tests else 0.0,
        'depth': saturate(build.sessions, 6),
    }
def rank(builds: list[BuildSignals], fits: list[float] | None = None, k: int = 3) -> list[ScoredBuild]:
    raw = fits if fits is not None else [0.0 for _ in builds]
    scaled = rescale(raw) if fits is not None else raw
    scored = []
    for build, fit, raw_fit in zip(builds, scaled, raw):
        parts = parts_for(build, fit)
        total = sum(WEIGHTS[name] * value for name, value in parts.items())
        scored.append(ScoredBuild(day=build.day, title=build.title, score=round(total, 4), role_fit=round(raw_fit, 4), parts=parts))
    return heapq.nlargest(k, scored, key=lambda item: (item.score, item.day))
def rank_all(builds: list[BuildSignals], fits: list[float] | None = None) -> list[ScoredBuild]:
    return rank(builds, fits, k=len(builds))
