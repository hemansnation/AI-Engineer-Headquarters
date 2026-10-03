import json
import time

from formats import FORMATS, shape
from llm import make_client, ask
from settings import Settings

MODEL = 'gemma'
QUESTION = 'Suggest a name for a coffee shop run by robots and say why.'

client = make_client()
calm = Settings(name='calm', temperature=0.2, max_output_tokens=200)

print('Part 1: same question, three shapes')
for style in FORMATS:
    result = ask(client, MODEL, shape(QUESTION, style), calm)
    print(f'[{style}] finish {result.finish} | out {result.output_tokens}')
    print(result.text)
    if style == 'json':
        try:
            json.loads(result.text)
            print('valid JSON: yes')
        except json.JSONDecodeError:
            print('valid JSON: no')
    time.sleep(1)

print()
print('Part 2: stop sequence, cut the list before item 3')
stopper = calm.tweak(name='stopper', stop_sequences=('3.',))
result = ask(client, MODEL, 'List 5 fruits as a numbered list.', stopper)
print(result.text)
print(f'finish {result.finish}')
time.sleep(1)

print()
print('Part 3: max tokens, cut it short on purpose')
tiny = calm.tweak(name='tiny', max_output_tokens=12)
result = ask(client, MODEL, 'Explain what an API is in three sentences.', tiny)
print(result.text)
print(f'finish {result.finish} | out {result.output_tokens}')