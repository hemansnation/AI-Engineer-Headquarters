import sys
from ollama import ResponseError
from llm import MODEL
from loop import run_assistant
def main():
    question = ' '.join(sys.argv[1:]) or 'What is 15 percent of 2480, plus 37 times 12?'
    try:
        answer = run_assistant(question)
    except ConnectionError:
        print(f'Ollama is not running. Start it, then run: ollama pull {MODEL}')
        sys.exit(1)
    except ResponseError as error:
        print(f'Ollama said: {error.error}')
        sys.exit(1)
    for call in answer.calls:
        status = 'ok' if call.ok else 'failed'
        print(f'{call.name}({call.arguments}) -> {call.content} [{status}, {call.seconds:.4f}s]')
    print(answer.text)
    print(f'rounds: {answer.rounds}  verified: {answer.verified}')
if __name__ == '__main__':
    main()
