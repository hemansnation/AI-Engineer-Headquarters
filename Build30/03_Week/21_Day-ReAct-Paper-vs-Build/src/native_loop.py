from llm import chat
from tracker import record
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'search',
        'description': 'Open a wiki page by its title and read the first sentences',
        'parameters': {'type': 'object', 'properties': {'entity': {'type': 'string', 'description': 'Page title'}}, 'required': ['entity']}}},
    {'type': 'function', 'function': {
        'name': 'lookup',
        'description': 'Find the next sentence with a keyword on the page opened last',
        'parameters': {'type': 'object', 'properties': {'keyword': {'type': 'string', 'description': 'Word to find'}}, 'required': ['keyword']}}},
    {'type': 'function', 'function': {
        'name': 'finish',
        'description': 'Give the short final answer and stop',
        'parameters': {'type': 'object', 'properties': {'answer': {'type': 'string', 'description': 'Final answer'}}, 'required': ['answer']}}},
]
SYSTEM = 'Answer the question by using the tools. When you know the answer, call finish with a short answer.'
def run(question, wiki, run_id, max_steps=7):
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': question}]
    for step in range(1, max_steps + 1):
        result = chat(messages, tools=TOOLS)
        record(run_id, step, result)
        calls = result['tool_calls']
        messages.append({'role': 'assistant', 'content': result['text'], 'tool_calls': calls})
        if not calls:
            return {'answer': result['text'], 'steps': step, 'stop': 'text'}
        for call in calls:
            name = call.function.name
            args = call.function.arguments
            if name == 'finish':
                return {'answer': str(args.get('answer', '')), 'steps': step, 'stop': 'finish'}
            if name == 'search':
                observation = wiki.search(str(args.get('entity', '')))
            elif name == 'lookup':
                observation = wiki.lookup(str(args.get('keyword', '')))
            else:
                observation = 'Unknown tool ' + name
            messages.append({'role': 'tool', 'content': observation, 'tool_name': name})
    return {'answer': '', 'steps': max_steps, 'stop': 'max_steps'}
