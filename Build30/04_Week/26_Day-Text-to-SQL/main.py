import sys
from guard import SQLRejected, validate
from db import run_query
ATTACKS = [
    'DROP TABLE orders',
    'SELECT id FROM orders; DELETE FROM orders',
    'UPDATE orders SET status = ' + chr(39) + 'delivered' + chr(39),
    'SELECT * FROM customers',
    'SELECT email FROM customers',
    'SELECT c.id FROM customers c WHERE c.phone LIKE ' + chr(39) + '+91%' + chr(39),
    'SELECT staff_name, monthly_salary_inr FROM staff_salaries',
    'SELECT name FROM sqlite_master',
    'PRAGMA writable_schema = 1',
    'ATTACH DATABASE ' + chr(39) + 'other.db' + chr(39) + ' AS other',
    'SELECT load_extension(' + chr(39) + 'evil' + chr(39) + ')',
    'SELECT id INTO backup FROM orders',
]
def explain_error(error):
    name = type(error).__name__
    if name == 'ConnectionError':
        return 'Ollama is not running. Open the Ollama app, or run ollama serve in another terminal.'
    if name == 'ResponseError':
        return 'Ollama answered with an error. If the model is missing, run ollama pull with the model name from your .env file.'
    return name + ': ' + str(error)
def red_team():
    blocked = 0
    for sql in ATTACKS:
        try:
            validate(sql)
            print('PASSED  ' + sql)
        except SQLRejected as error:
            blocked += 1
            print('BLOCKED ' + sql + '  ->  ' + str(error))
    print(str(blocked) + ' of ' + str(len(ATTACKS)) + ' attacks blocked by the validator.')
    for sql in ['DELETE FROM orders', 'SELECT email FROM customers']:
        try:
            run_query(sql)
            print('DATABASE ALLOWED ' + sql)
        except Exception as error:
            print('DATABASE REFUSED ' + sql + '  ->  ' + str(error))
def ask_command(question):
    from pipeline import ask
    result = ask(question)
    print('Tables picked: ' + ', '.join(name + ' ' + str(score) for name, score in result['tables']))
    for item in result['attempts']:
        print('Rejected: ' + item['rejected'])
    if result['sql']:
        print('SQL: ' + result['sql'])
        if result.get('assumption'):
            print('Assumption: ' + result['assumption'])
        print(result['result'])
    print('Answer: ' + result['answer'])
if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    try:
        if command == 'build-db':
            from data.build_db import build
            print('Database written to ' + str(build()))
        elif command == 'redteam':
            red_team()
        elif command == 'ask' and len(sys.argv) > 2:
            ask_command(' '.join(sys.argv[2:]))
        else:
            print('Use: python main.py build-db | python main.py redteam | python main.py ask your question')
    except Exception as error:
        sys.exit(explain_error(error))
