## react agent loop from scratch

main.py > agent/loop.py (thought, action, observation, max 6 steps) > llm.py (ollama, stop on Observation) > parser.py > tools.py (calculator, count_letter, search_notes) > guard.py + tracker.py
tests use a scripted fake model, no ollama needed: python -m pytest -q tests

run: ollama pull qwen3:4b, then python main.py
