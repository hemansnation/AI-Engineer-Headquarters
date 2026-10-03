import argparse
import time

from rich.console import Console
from rich.table import Table

from llm import MODELS, make_client, ask
from settings import PRESETS

HINTS = {
    400: 'the model rejected a setting, try removing temperature or top_p',
    404: 'wrong model id, check MODELS in llm.py',
    429: 'rate limit on the free tier, raise --pause or lower --runs',
}

DEFAULT_PROMPT = 'Give me one name for a coffee shop run by robots. Only the name.'

def explain(error):
    code = getattr(error, 'code', None)
    hint = HINTS.get(code, 'read the error message and search the status code')
    return f'ERROR {code}: {hint}'

def unique_count(results):
    good = [r.text.lower() for r in results if not isinstance(r, str)]
    return len(set(good))

def verdict(unique_by_preset):
    low = unique_by_preset.get('precise')
    high = unique_by_preset.get('wild')
    if low is None or high is None:
        return 'Add precise and wild to --presets to get a verdict.'
    if low == 0 and high == 0:
        return 'Every call failed, fix the error first.'
    gap = high - low
    if gap >= 2:
        return 'Knob looks alive: wild gave clearly more different answers than precise.'
    if gap == 1:
        return 'Small gap, this could be luck. Run again with --runs 8.'
    return (
        'No visible difference between precise and wild. Either this model '
        'ignores temperature, or your prompt leaves no room to vary.'
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', nargs='*')
    parser.add_argument('--model', default='gemma', choices=list(MODELS))
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--presets', default='precise,balanced,wild')
    parser.add_argument('--pause', type=float, default=2.0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    prompt = ' '.join(args.prompt) or DEFAULT_PROMPT
    names = [n.strip() for n in args.presets.split(',')]
    client = make_client()
    console = Console()

    columns = {}
    unique_by_preset = {}
    total_tokens = 0

    for name in names:
        settings = PRESETS[name]
        results = []
        for _ in range(args.runs):
            try:
                result = ask(client, args.model, prompt, settings, force=args.force)
                total_tokens += result.input_tokens + result.output_tokens + result.thought_tokens
                results.append(result)
            except Exception as error:
                results.append(explain(error))
            time.sleep(args.pause)
        columns[name] = results
        unique_by_preset[name] = unique_count(results)

    table = Table(title=f'{args.model} | {args.runs} runs | {prompt}', show_lines=True)
    for name in names:
        table.add_column(name, overflow='fold')

    for i in range(args.runs):
        row = []
        for name in names:
            item = columns[name][i]
            row.append(item if isinstance(item, str) else item.text)
        table.add_row(*row)

    summary = []
    for name in names:
        good = [r for r in columns[name] if not isinstance(r, str)]
        avg = round(sum(r.output_tokens for r in good) / len(good)) if good else 0
        finishes = sorted({r.finish for r in good})
        summary.append(f'unique {unique_by_preset[name]}/{args.runs}\navg out {avg}\n{finishes}')
    table.add_row(*summary)
    console.print(table)

    for name in names:
        good = [r for r in columns[name] if not isinstance(r, str)]
        if good and good[0].dropped:
            console.print(f'{name}: sent without {good[0].dropped} for this model')

    console.print(verdict(unique_by_preset))
    console.print(f'total tokens used: {total_tokens}')

if __name__ == '__main__':
    main()