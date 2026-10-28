from ollama import Client
from config import OLLAMA_HOST, OLLAMA_MODEL
from schemas import SQLDraft
from tracker import track_tokens
client = Client(host=OLLAMA_HOST, timeout=300)
SQL_SYSTEM = ('You write SQLite queries for a support team. '
    'Use only the tables and columns listed below. Never use SELECT *. '
    'Write one SELECT statement. Never write INSERT, UPDATE, DELETE, DROP or any other change. '
    'Dates are text in YYYY-MM-DD format. Return JSON only.')
ANSWER_SYSTEM = ('You answer a business question using only the query result below. '
    'If the result is empty, say that no matching rows were found. '
    'If the result says it was cut off, say the numbers cover only the rows shown. '
    'Never invent a number that is not in the result. Answer in 2 or 3 sentences.')
@track_tokens
def write_sql_call(messages):
    return client.chat(model=OLLAMA_MODEL, messages=messages, format=SQLDraft.model_json_schema(), options={'temperature': 0}, think=False)
@track_tokens
def answer_call(messages):
    return client.chat(model=OLLAMA_MODEL, messages=messages, options={'temperature': 0}, think=False)
def start_messages(question, schema_text):
    return [{'role': 'system', 'content': SQL_SYSTEM + '\n' + schema_text}, {'role': 'user', 'content': question}]
def write_sql(messages):
    text = write_sql_call(messages).message.content
    messages.append({'role': 'assistant', 'content': text})
    return SQLDraft.model_validate_json(text)
def give_feedback(messages, reason):
    messages.append({'role': 'user', 'content': 'That query was not run. Reason: ' + reason + ' Write a corrected query.'})
def answer(question, sql, result_text):
    messages = [{'role': 'system', 'content': ANSWER_SYSTEM}, {'role': 'user', 'content': 'Question: ' + question + '\nSQL: ' + sql + '\nResult:\n' + result_text}]
    return answer_call(messages).message.content.strip()
