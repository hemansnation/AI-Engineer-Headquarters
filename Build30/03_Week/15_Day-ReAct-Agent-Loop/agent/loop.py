import time
from agent.llm import chat
from agent.parser import parse
from agent.tools import TOOLS
from agent.tracker import TokenTracker
from agent.guard import LoopGuard
MAX_ITERATIONS = 6
MAX_OBSERVATION_CHARS = 1500
def build_system_prompt(tools):
    lines = []
    for name, info in tools.items():
        lines.append('- ' + name + ': ' + info['description'])
    return (
        'You answer questions by using tools. Use this exact format.\n'
        'Thought: say what you need to do next\n'
        'Action: one tool name from the list\n'
        'Action Input: the input for that tool on one line\n'
        'Then stop and wait. You will get an Observation.\n'
        'Repeat Thought, Action, Action Input until you know the answer.\n'
        'When you know it, write:\n'
        'Thought: I know the answer\n'
        'Final Answer: your answer in one or two sentences\n'
        'Never write Observation yourself. Never guess what a tool returns.\n'
        'Tools:\n' + '\n'.join(lines)
    )
def clip(text, limit=MAX_OBSERVATION_CHARS):
    if len(text) <= limit:
        return text
    return text[:limit] + ' ...[cut, ' + str(len(text) - limit) + ' more characters]'
def run_tool(name, tool_input):
    if name not in TOOLS:
        return 'Error: no tool called ' + name + '. Pick one of: ' + ', '.join(TOOLS)
    try:
        return clip(TOOLS[name]['func'](tool_input))
    except Exception as error:
        return 'Error: ' + type(error).__name__ + ': ' + str(error)
def run_agent(question, llm=chat, max_iterations=MAX_ITERATIONS):
    messages = [
        {'role': 'system', 'content': build_system_prompt(TOOLS)},
        {'role': 'user', 'content': 'Question: ' + question},
    ]
    tracker = TokenTracker()
    guard = LoopGuard()
    trace = []
    for step in range(1, max_iterations + 1):
        started = time.perf_counter()
        reply = llm(messages, stop=['Observation:'])
        tracker.record(step, reply['prompt_tokens'], reply['completion_tokens'])
        parsed = parse(reply['text'])
        messages.append({'role': 'assistant', 'content': reply['text']})
        seconds = round(time.perf_counter() - started, 2)
        if parsed['kind'] == 'final':
            trace.append({'step': step, 'thought': parsed['thought'], 'action': None, 'input': None, 'observation': None, 'seconds': seconds})
            return {'answer': parsed['answer'], 'status': 'finished', 'steps': step, 'trace': trace, 'tokens': tracker}
        if parsed['kind'] == 'invalid':
            action, tool_input = None, None
            observation = 'Error: I could not find an Action or a Final Answer. Use the exact format.'
        else:
            action, tool_input = parsed['tool'], parsed['input']
            observation = guard.check(action, tool_input) or run_tool(action, tool_input)
        trace.append({'step': step, 'thought': parsed['thought'], 'action': action, 'input': tool_input, 'observation': observation, 'seconds': seconds})
        messages.append({'role': 'user', 'content': 'Observation: ' + observation})
    messages[-1]['content'] += '\nYou are out of steps. Write the Final Answer now with what you know.'
    reply = llm(messages, stop=['Observation:'])
    tracker.record(max_iterations + 1, reply['prompt_tokens'], reply['completion_tokens'])
    parsed = parse(reply['text'])
    answer = parsed['answer'] if parsed['kind'] == 'final' else reply['text'].strip()
    return {'answer': answer, 'status': 'hit_cap', 'steps': max_iterations, 'trace': trace, 'tokens': tracker}
