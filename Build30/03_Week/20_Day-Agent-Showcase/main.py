import sys
from agent import run_agent
def main():
    task = ' '.join(sys.argv[1:]) or 'What is 17 * 23?'
    record = run_agent(task)
    for step in record['steps']:
        print('[' + str(step['i']) + '] ' + step['kind'] + ': ' + step['text'])
    print('status: ' + record['status'])
    print('answer: ' + record['final'])
    print('flags: ' + ', '.join(record['flags']))
    print('tokens: ' + str(record['prompt_tokens']) + ' in, ' + str(record['completion_tokens']) + ' out')
    print('latency: ' + str(record['latency_s']) + ' s, run id ' + record['run_id'])
if __name__ == '__main__':
    main()
