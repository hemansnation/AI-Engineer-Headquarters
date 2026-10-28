import re
import time
from config import SYSTEM_PROMPT, MAX_ITERATIONS, OLLAMA_MODEL
from llm import chat
from tools import run_tool
from guardrail import scan_input, mask_pii_entities, check_output
from recorder import append_run, LEDGER
ACTION = re.compile(r'^ACTION:\s*(\w+)\s*\|\s*(.*)$', re.MULTILINE)
FINAL = re.compile(r'^FINAL:\s*(.*)$', re.MULTILINE | re.DOTALL)
def parse(text):
    action, final = ACTION.search(text), FINAL.search(text)
    if action and (not final or action.start() < final.start()):
        return 'action', action.group(1), action.group(2).strip()
    if final:
        return 'final', final.group(1).strip(), ''
    return 'none', text.strip(), ''
def run_agent(task, ledger=LEDGER, chat_fn=chat):
    start = time.perf_counter()
    steps, flags = [], []
    tokens = {'prompt': 0, 'completion': 0}
    logged_task = mask_pii_entities(task)[0]
    verdict = scan_input(task)
    if not verdict['ok']:
        status = 'blocked_input'
        final = 'Request blocked by the input guardrail.'
        flags += verdict['reasons']
    else:
        status = 'max_iterations'
        final = 'Stopped after the iteration cap without a final answer.'
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': logged_task}]
        for i in range(MAX_ITERATIONS):
            try:
                text, usage = chat_fn(messages)
            except Exception as error:
                status, final = 'error', 'Model call failed: ' + type(error).__name__
                break
            tokens['prompt'] += usage['prompt']
            tokens['completion'] += usage['completion']
            kind, first, second = parse(text)
            steps.append({'i': i, 'kind': kind, 'text': text[:300]})
            messages.append({'role': 'assistant', 'content': text})
            if kind == 'final':
                verdict_out = check_output(first)
                masked, counts = mask_pii_entities(first)
                if verdict_out['ok']:
                    status, final = 'ok', masked
                    if counts:
                        flags.append('pii_masked_in_output')
                else:
                    status = 'blocked_output'
                    final = 'Response blocked by the output guardrail.'
                    flags += verdict_out['reasons']
                break
            if kind == 'action':
                observation = run_tool(first, second)
                seen = scan_input(observation)
                if not seen['ok']:
                    flags += ['indirect_injection_quarantined'] + seen['reasons']
                    observation = 'BLOCKED: this tool result contained instructions and was removed.'
                else:
                    observation, counts = mask_pii_entities(observation)
                    if counts:
                        flags.append('pii_masked_in_tool_result')
                steps.append({'i': i, 'kind': 'observation', 'text': observation[:300]})
                messages.append({'role': 'user', 'content': 'OBSERVATION: ' + observation})
            else:
                messages.append({'role': 'user', 'content': 'Reply with one line that starts with ACTION: or FINAL:'})
    record = {
        'task': logged_task,
        'status': status,
        'final': final,
        'steps': steps,
        'flags': sorted(set(flags)),
        'latency_s': round(time.perf_counter() - start, 3),
        'prompt_tokens': tokens['prompt'],
        'completion_tokens': tokens['completion'],
        'model': OLLAMA_MODEL,
    }
    return append_run(record, ledger) if ledger else record
