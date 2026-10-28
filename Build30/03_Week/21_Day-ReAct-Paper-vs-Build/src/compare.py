import re
from pathlib import Path
import llm
import mine_loop
import native_loop
import paper_loop
from wiki import Wiki
ROOT = Path(__file__).resolve().parent.parent
def load_questions():
    rows = []
    for line in (ROOT / 'data' / 'questions.txt').read_text(encoding='utf-8').splitlines():
        if ' => ' in line:
            question, gold = line.split(' => ', 1)
            rows.append((question.strip(), gold.strip()))
    return rows
def normalize(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()
def measure(fn):
    before = dict(llm.USAGE)
    result = fn()
    result['calls'] = llm.USAGE['calls'] - before['calls']
    result['tokens'] = llm.USAGE['input'] + llm.USAGE['output'] - before['input'] - before['output']
    result['seconds'] = round(llm.USAGE['seconds'] - before['seconds'], 1)
    return result
def main():
    wiki = Wiki()
    registry = mine_loop.build_registry(wiki)
    rows = load_questions()
    runners = {
        'paper': lambda question, number: paper_loop.run(question, wiki),
        'mine': lambda question, number: mine_loop.run(question, registry, 'mine-' + str(number)),
        'native': lambda question, number: native_loop.run(question, wiki, 'native-' + str(number)),
    }
    table = [
        '| loop | question | answer | correct | steps | calls | tokens | seconds | stop |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    scores = {name: 0 for name in runners}
    traces = []
    for number, (question, gold) in enumerate(rows, start=1):
        for name, runner in runners.items():
            wiki.reset()
            result = measure(lambda: runner(question, number))
            correct = normalize(gold) in normalize(result['answer'])
            scores[name] += int(correct)
            if name == 'paper':
                traces.append('Question: ' + question + '\n' + result['scratchpad'])
            row = [name, 'Q' + str(number), result['answer'] or '-', 'yes' if correct else 'no', str(result['steps']), str(result['calls']), str(result['tokens']), str(result['seconds']), result['stop']]
            table.append('| ' + ' | '.join(row) + ' |')
            print(table[-1])
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    score_lines = [name + ': ' + str(score) + ' of ' + str(len(rows)) + ' correct' for name, score in scores.items()]
    report = '# Day 21 comparison report\n\n' + '\n'.join(table) + '\n\n' + '\n'.join(score_lines) + '\n'
    (out / 'report.md').write_text(report, encoding='utf-8')
    (out / 'paper_traces.txt').write_text('\n'.join(traces), encoding='utf-8')
    print('\n'.join(score_lines))
if __name__ == '__main__':
    main()
