def compare_runs(base, new):
    if base['golden_hash'] != new['golden_hash']:
        a, b = base['golden_hash'], new['golden_hash']
        raise ValueError(f'golden sets differ: {a} vs {b}')
    b = {r['id']: r for r in base['results']}
    n = {r['id']: r for r in new['results']}
    fixed = sorted(i for i in b if not b[i]['passed'] and n[i]['passed'])
    broke = sorted(i for i in b if b[i]['passed'] and not n[i]['passed'])
    return {'fixed': fixed, 'broke': broke, 'delta': round(new['pass_rate'] - base['pass_rate'], 4), 'broke_must_pass': [i for i in broke if n[i]['must_pass']]}
def by_tag(run):
    tags = {}
    for r in run['results']:
        for t in r['tags'] or ['untagged']:
            tags.setdefault(t, []).append(r['passed'])
    return {t: f'{sum(v)}/{len(v)}' for t, v in sorted(tags.items())}
def format_run(run):
    label, passed, total = run['label'], run['passed'], run['total']
    pct, secs, use = run['pass_rate'] * 100, run['seconds'], run['usage']
    tokens_in, tokens_out = use['prompt_tokens'], use['output_tokens']
    lines = [f'{label}: {passed}/{total} passed ({pct:.1f}%) in {secs}s, tokens {tokens_in} in, {tokens_out} out']
    lines.append('by tag: ' + ', '.join(f'{t} {v}' for t, v in by_tag(run).items()))
    for r in run['results']:
        if not r['passed']:
            rid, scorer, got, note = r['id'], r['scorer'], r['output'][:70], r['note'][:60]
            lines.append(f'  FAIL {rid} [{scorer}] got: {got} | {note}')
    return '\n'.join(lines)
def format_compare(base, new, diff):
    a, b = base['label'], new['label']
    pa = str(base['passed']) + '/' + str(base['total'])
    pb = str(new['passed']) + '/' + str(new['total'])
    points = diff['delta'] * 100
    lines = [f'{a} {pa} -> {b} {pb} (delta {points:+.1f} points)']
    lines.append('fixed: ' + str(diff['fixed']))
    lines.append('broke: ' + str(diff['broke']))
    return '\n'.join(lines)
