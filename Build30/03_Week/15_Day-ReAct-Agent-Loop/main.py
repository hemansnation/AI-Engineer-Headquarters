import sys
from agent.loop import run_agent
DEFAULT_QUESTION = 'How many r letters are in strawberry, and what is that number times 12?'
def show(result):
    for item in result['trace']:
        print('Step', item['step'])
        print('  Thought     :', item['thought'])
        if item['action']:
            print('  Action      :', item['action'], '(' + str(item['input']) + ')')
            print('  Observation :', item['observation'])
    print('Status:', result['status'], 'in', result['steps'], 'steps')
    print('Answer:', result['answer'])
    print(result['tokens'].table())
def main():
    question = ' '.join(sys.argv[1:]) or DEFAULT_QUESTION
    result = run_agent(question)
    show(result)
    result['tokens'].save('logs/token_log.jsonl', question, result['status'])
if __name__ == '__main__':
    main()
