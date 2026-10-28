# llmcost

LLM cost modeling tool: reasoning tokens, prompt-cache economics, agent Monte Carlo (p50/p90/p99),
self-host vs API breakeven, prompt cost linter, unit economics. Specs in `docs/`.

## Rules
- Python >=3.12, free/open-source dependencies only. Ask before adding any dependency.
- `engine/` and `sim/` are pure functions: no I/O, no web, no DB. API, CLI, UI are thin shells.
- Money is `decimal.Decimal`, never float. Prices are USD per 1M tokens.
- Never hard-code prices in Python. All prices live in `data/pricing/*.yaml` with `source_url` and `last_verified`.
- Formulas are specified in `docs/formulas.md`; implement exactly and test each with a hand-computed golden case (arithmetic in a comment).
- Every new function gets tests. Before saying done run: `pytest -q`, `ruff check .`, `mypy src`. Show the output.
- Keep scope tight: only touch files the task names.

## Commands
- Setup: `python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- Test: `.venv/bin/pytest -q --cov=llmcost`
- Lint/types: `.venv/bin/ruff check . && .venv/bin/mypy src`
