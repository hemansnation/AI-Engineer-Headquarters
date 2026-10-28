import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from config import MAX_TRIES
from schema_cards import card_text
from guard import SQLRejected, validate
from db import run_query
from shaper import to_table
AUDIT = Path('output/audit.jsonl')
def audit(record):
    AUDIT.parent.mkdir(exist_ok=True)
    record['time'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with AUDIT.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record) + '\n')
def ask(question, model=None):
    import llm
    from retriever import pick_tables
    picked = pick_tables(question, model=model)
    schema_text = '\n'.join(card_text(name) for name, _ in picked)
    messages = llm.start_messages(question, schema_text)
    attempts = []
    for attempt in range(1, MAX_TRIES + 1):
        draft = llm.write_sql(messages)
        try:
            safe_sql = validate(draft.sql)
            result = run_query(safe_sql)
        except (SQLRejected, sqlite3.Error, TimeoutError) as error:
            reason = str(error)
            attempts.append({'sql': draft.sql, 'rejected': reason})
            audit({'question': question, 'attempt': attempt, 'sql': draft.sql, 'status': 'rejected', 'reason': reason})
            llm.give_feedback(messages, reason)
            continue
        audit({'question': question, 'attempt': attempt, 'sql': safe_sql, 'status': 'ran', 'rows': len(result.rows), 'truncated': result.truncated, 'ms': result.milliseconds})
        table_text = to_table(result)
        return {'tables': picked, 'sql': safe_sql, 'assumption': draft.assumption, 'result': table_text, 'answer': llm.answer(question, safe_sql, table_text), 'attempts': attempts}
    audit({'question': question, 'status': 'gave_up'})
    return {'tables': picked, 'sql': None, 'answer': 'I could not write a safe query for this question after ' + str(MAX_TRIES) + ' tries. Try asking it more narrowly.', 'attempts': attempts}
