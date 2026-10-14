import argparse
import sys
from pathlib import Path

from extractor import extract, log_result, read_totals
from llm import get_provider

def main():
    parser = argparse.ArgumentParser(description='Day 11 structured extraction')
    parser.add_argument('--file', default='samples/messy_email.txt')
    parser.add_argument('--provider', default='gemini', choices=['gemini', 'ollama', 'fake'])
    parser.add_argument('--mode', default='native', choices=['native', 'prompt_only'])
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()

    if args.summary:
        totals = read_totals()
        print('calls:', totals['calls'])
        print('succeeded:', totals['ok'])
        print('total attempts:', totals['attempts'])
        print('total tokens:', totals['tokens'])
        return

    email_text = Path(args.file).read_text(encoding='utf-8')
    provider = get_provider(args.provider)
    result = extract(email_text, provider, args.mode, args.max_attempts)
    log_result(result, args.provider, args.mode)

    print('=' * 50)
    print('provider:', args.provider, '| mode:', args.mode)
    print('attempts:', result.attempts)
    print('tokens in/out:', result.input_tokens, '/', result.output_tokens)
    for number, problem in enumerate(result.errors, start=1):
        print('rejected attempt', number, '->', problem.replace('\n', ' | '))
    print('=' * 50)

    if result.ok:
        print(result.data.model_dump_json(indent=2))
    else:
        print('FAILED after', result.attempts, 'attempts. Last raw output:')
        print(result.last_raw)
        sys.exit(1)

if __name__ == '__main__':
    main()
