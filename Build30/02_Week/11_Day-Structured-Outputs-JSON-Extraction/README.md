## structured outputs and json extraction

main.py > extractor.py (prompt, validate, retry with exact errors, max 3) > schemas.py (pydantic contract) > llm.py (gemini, ollama, fake)
tests run with the fake provider, no key needed: python -m pytest -q tests

try: python main.py --provider fake --file samples/messy_email.txt
