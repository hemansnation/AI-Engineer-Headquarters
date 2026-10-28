# Product scope

Know what an LLM feature will cost before you ship it, catch cost regressions in PRs, and see real
cost per customer after you ship.

MVP (phases 0-4): pricing registry (source + last_verified), request cost engine (reasoning, cache,
batch), workload projection, cache breakeven, self-host breakeven, agent Monte Carlo, API + UI.
Stickiness (5-6): CLI, prompt linter, GitHub Action cost diff, usage ingestion, unit economics.

Not in scope: production proxy, semantic caching, multi-tenant auth/billing, live price scraping.

Success: estimate within 5% of invoice; 10k-run sim of 10-turn agent under 2s; every price shows source
and last-verified (warn if >30 days); 90%+ coverage on `engine/` and `sim/`.

Pricing honesty rule: no prices in Python; all in data files.
