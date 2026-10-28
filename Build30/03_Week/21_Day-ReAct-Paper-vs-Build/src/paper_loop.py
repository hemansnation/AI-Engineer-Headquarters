import re
from llm import chat
from prompts import paper_prompt_start
ACTION = re.compile(r'Action\s*\d*:\s*(\w+)\[(.*)\]')
def run(question, wiki, max_steps=7):
    scratchpad = ''
    for step in range(1, max_steps + 1):
        prompt = paper_prompt_start() + 'Question: ' + question + '\n' + scratchpad + 'Thought ' + str(step) + ':'
        reply = chat([{'role': 'user', 'content': prompt}], stop=['\nObservation'])['text']
        scratchpad += 'Thought ' + str(step) + ': ' + reply + '\n'
        match = ACTION.search(reply)
        if not match:
            scratchpad += 'Observation ' + str(step) + ': Invalid action. Use Search[entity], Lookup[keyword] or Finish[answer].\n'
            continue
        name, argument = match.group(1), match.group(2).strip()
        if name == 'Finish':
            return {'answer': argument, 'steps': step, 'stop': 'finish', 'scratchpad': scratchpad}
        if name == 'Search':
            observation = wiki.search(argument)
        elif name == 'Lookup':
            observation = wiki.lookup(argument)
        else:
            observation = 'Unknown action ' + name + '. Use Search, Lookup or Finish.'
        scratchpad += 'Observation ' + str(step) + ': ' + observation + '\n'
    return {'answer': '', 'steps': max_steps, 'stop': 'max_steps', 'scratchpad': scratchpad}
