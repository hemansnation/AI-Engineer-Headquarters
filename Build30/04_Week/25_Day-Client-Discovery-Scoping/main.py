import json
import sys
from pathlib import Path
from config import EVIDENCE_THRESHOLD, HOURLY_RATE_USD, OVERHEAD_PCT, RISK_HOLDBACK
from schemas import Brief
from scoping import load_features, pick_features
from pricing import price_scope, run_cost
from gate import run_gate
from report import build_report
TRANSCRIPT = Path('data/discovery_call.txt')
FEATURES = Path('data/features.csv')
BRIEF = Path('output/brief.json')
EVIDENCE = Path('output/evidence.json')
SCOPE = Path('output/scope.md')
def explain_error(error):
    name = type(error).__name__
    if name == 'ConnectionError':
        return 'Ollama is not running. Open the Ollama app, or run ollama serve in another terminal.'
    if name == 'ResponseError':
        return 'Ollama answered with an error. If the model is missing, run ollama pull with the model name from your .env file.'
    return name + ': ' + str(error)
def draft():
    from llm import draft_brief
    from evidence import best_evidence, check_numbers
    from tracker import running_totals
    transcript = TRANSCRIPT.read_text(encoding='utf-8')
    brief = draft_brief(transcript)
    fields = {'stated_problem': brief.stated_problem, 'underlying_problem': brief.underlying_problem, 'current_process': brief.current_process, 'baseline_metric': brief.baseline_metric, 'success_metric': brief.success_metric}
    claims = {name: text for name, text in fields.items() if text}
    evidence = best_evidence(claims, transcript, EVIDENCE_THRESHOLD)
    missing = check_numbers(brief, transcript)
    BRIEF.parent.mkdir(exist_ok=True)
    BRIEF.write_text(brief.model_dump_json(indent=2), encoding='utf-8')
    EVIDENCE.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('Draft brief for ' + brief.client)
    for name, found in evidence.items():
        flag = 'ok  ' if found['grounded'] else 'WEAK'
        score = found['score']
        print(flag + ' ' + name + ' (score ' + str(score) + '): ' + found['line'][:70])
    for name in missing:
        print('NUMBER NOT IN CALL: ' + name)
    totals = running_totals()
    print('Model calls so far: ' + str(totals['calls']) + ', tokens: ' + str(totals['tokens']))
    print('Now open output/brief.json, fix anything wrong, then run: python main.py scope')
def scope():
    brief = Brief.model_validate_json(BRIEF.read_text(encoding='utf-8'))
    if brief.budget_usd is None:
        sys.exit('The brief has no budget. Ask the client, add budget_usd to output/brief.json, then run again.')
    evidence = json.loads(EVIDENCE.read_text(encoding='utf-8')) if EVIDENCE.exists() else {}
    budget_hours = brief.budget_usd / (HOURLY_RATE_USD * (1 + OVERHEAD_PCT))
    capacity = int(budget_hours * (1 - RISK_HOLDBACK))
    chosen, later = pick_features(load_features(FEATURES), capacity)
    hours, prices = price_scope(chosen)
    running = run_cost(brief.volume_per_month or 0, brief.automatable_share or 0)
    gate = run_gate(brief, prices['p80'], running['hosting_usd'])
    SCOPE.write_text(build_report(brief, chosen, later, hours, prices, running, gate, evidence), encoding='utf-8')
    print('Verdict: ' + gate['verdict'])
    print('Quote: $' + str(prices['p80']) + ', payback in months: ' + str(gate['payback_months']))
    print('Scope written to output/scope.md')
if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    try:
        if command == 'draft':
            draft()
        elif command == 'scope':
            scope()
        else:
            print('Use: python main.py draft   or   python main.py scope')
    except Exception as error:
        sys.exit(explain_error(error))
