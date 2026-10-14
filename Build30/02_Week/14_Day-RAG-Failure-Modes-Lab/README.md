## rag failure lab

main.py > naive pipeline vs fixed pipeline (7 flags in pipeline.py) > report.py table, one number per failure
confident answer with no evidence, lost in the middle, chunk boundary loss, retrieval drift
logs/run_log_*.jsonl saved for triage. llm_probe.py is optional (needs Gemini key)

run: python main.py
