from data.samples import CLEAN_PROMPTS
from guard.pipeline import build_assistant
def average_ms(assistant):
    totals = [assistant.handle(text)['timings']['total_ms'] for text in CLEAN_PROMPTS]
    return round(sum(totals) / len(totals), 1)
no_guard_llm = average_ms(build_assistant(use_judge=False))
sequential = average_ms(build_assistant(use_judge=True, parallel=False))
parallel = average_ms(build_assistant(use_judge=True, parallel=True))
print('no guard LLM  ', no_guard_llm, 'ms')
print('dual LLM seq  ', sequential, 'ms', 'extra', round(sequential - no_guard_llm, 1), 'ms')
print('dual LLM para ', parallel, 'ms', 'extra', round(parallel - no_guard_llm, 1), 'ms')
