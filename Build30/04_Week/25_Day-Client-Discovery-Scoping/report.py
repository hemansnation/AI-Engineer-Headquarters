from datetime import date
def assumptions(brief):
    items = [f'Access to {source} is ready by the end of week 1.' for source in brief.data_sources]
    items += ['The client reviews each draft within 2 working days.', 'The quote comes from three-point estimates plus a risk buffer. Re-check it after week 2.', 'Anything outside the in-scope list is a change request with its own hours and price.']
    return items
def build_report(brief, chosen, later, hours, prices, running, gate, evidence):
    lines = [f'# Scope proposal for {brief.client}', f'Date: {date.today().isoformat()}']
    lines += ['## Stated problem', brief.stated_problem, '## Underlying problem', brief.underlying_problem, '## Verdict', gate['verdict']]
    lines += [f'- Blocker: {text}' for text in gate['blockers']]
    lines += [f'- Condition: {text}' for text in gate['conditions']]
    lines += ['## In scope for phase 1']
    lines += [f'- {f.name} (value {f.value}, likely {f.likely} hours)' for f in chosen]
    lines += ['## Not in phase 1']
    lines += [f'- {f.name} (value {f.value}, likely {f.likely} hours)' for f in later]
    lines += ['## Price']
    lines += ['- Expected effort: ' + format(hours['expected'], '.0f') + ' hours']
    lines += ['- Quote (expected hours plus a risk buffer): $' + str(prices['p80']), '- Worst case (bigger buffer): $' + str(prices['p95']), '- Price with no buffer: $' + str(prices['expected'])]
    lines += ['## Running cost and payback']
    lines += ['- Tasks handled per month: ' + str(running['tasks_per_month']), '- Tokens per month: ' + str(running['tokens_per_month']), '- Peak requests per minute: ' + str(running['peak_requests_per_minute']), '- Hosting per month: $' + format(running['hosting_usd'], '.0f')]
    lines += ['- Monthly saving in agent time: $' + str(gate['monthly_saving_usd']), '- Payback in months: ' + str(gate['payback_months'])]
    lines += ['## Assumptions']
    lines += [f'- {text}' for text in assumptions(brief)]
    lines += ['## Open questions']
    lines += [f'- {text}' for text in brief.open_questions] or ['- None']
    if evidence:
        lines += ['## Where each claim came from']
        for name, found in evidence.items():
            flag = 'ok' if found['grounded'] else 'WEAK, confirm on the next call'
            lines += ['- ' + name + ' (' + flag + ', score ' + str(found['score']) + '): ' + found['line']]
    return '\n'.join(lines) + '\n'
