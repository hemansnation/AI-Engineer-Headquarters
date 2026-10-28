import os
import re
from llm import chat
from prompts import mine_system
from registry import Registry
from tracker import record
ACTION = re.compile(r'Action\s*\d*:\s*(\w+)\[(.*)\]')
def build_registry(wiki):
    registry = Registry()
    registry.register('Search', 'open a page by its title and read the first sentences, or get similar titles')(wiki.search)
    registry.register('Lookup', 'find the next sentence with a keyword on the page you opened last')(wiki.lookup)
    return registry
def run(question, registry, run_id):
    max_steps = int(os.getenv('MAX_STEPS', '7'))
    messages = [
        {'role': 'system', 'content': mine_system(registry)},
        {'role': 'user', 'content': 'Question: ' + question},
    ]
    seen = set()
    for step in range(1, max_steps + 1):
        result = chat(messages, stop=['\nObservation'])
        record(run_id, step, result)
        reply = result['text']
        messages.append({'role': 'assistant', 'content': reply})
        match = ACTION.search(reply)
        if not match:
            messages.append({'role': 'user', 'content': 'Observation: Invalid reply. Write one Thought and one Action.'})
            continue
        name, argument = match.group(1), match.group(2).strip()
        if name == 'Finish':
            return {'answer': argument, 'steps': step, 'stop': 'finish'}
        if (name, argument) in seen:
            observation = 'You already did this exact action. Try something different or Finish.'
        else:
            seen.add((name, argument))
            observation = registry.run(name, argument)
        messages.append({'role': 'user', 'content': 'Observation: ' + observation})
    return {'answer': '', 'steps': max_steps, 'stop': 'max_steps'}
