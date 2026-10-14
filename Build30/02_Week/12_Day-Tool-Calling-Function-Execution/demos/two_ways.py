import ast
from llm import MODEL, client
from registry import registry
import tools
QUESTION = 'What is 15 percent of 2480, plus 37 times 12?'
print('WAY ONE: ask for code in plain text')
plain = client.chat(
    model=MODEL,
    messages=[{'role': 'user', 'content': QUESTION + ' Write Python code that prints the answer. Reply with code only.'}],
    think=False,
    options={'temperature': 0},
)
text = plain.message.content
print(text)
try:
    ast.parse(text)
    print('This text happens to be valid Python. I still would never run it.')
except SyntaxError:
    print('This text is not valid Python. Chat words or markdown marks got mixed in.')
print()
print('WAY TWO: native function calling')
native = client.chat(
    model=MODEL,
    messages=[{'role': 'user', 'content': QUESTION}],
    tools=registry.schemas(),
    think=False,
    options={'temperature': 0},
)
for call in native.message.tool_calls or []:
    print(call.function.name, call.function.arguments)
print('Nothing ran yet. The model only asked. My code decides what happens next.')
