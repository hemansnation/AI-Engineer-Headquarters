## tool calling and function execution

main.py > loop.py (round trip, round cap, answer check) > llm.py (ollama) > registry.py (allowlist, arg and type checks, approval gate) > tools.py (calculate, get_stock_price)
safe_math.py reads math with ast, never eval

no API key, local model: ollama pull qwen3:8b
tests need no model: python -m pytest -q
