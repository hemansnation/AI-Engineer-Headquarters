import math
import re
from dataclasses import dataclass
import tools
from llm import chat
from registry import registry
SYSTEM_PROMPT = '''You are a math and stock assistant.
Never do arithmetic in your head. Call calculate for every calculation.
Call get_stock_price for every stock price. Never guess a price.
If a tool returns an Error, read it, fix your arguments and try again, or tell the user what went wrong.
When you have the final number, state it clearly in one or two sentences.'''
NUMBER_PATTERN = re.compile(r'-?\d[\d,]*\.?\d*')
@dataclass
class Answer:
    text: str
    verified: bool
    calls: list
    rounds: int
def numbers_in(text):
    found = []
    for piece in NUMBER_PATTERN.findall(text):
        try:
            found.append(float(piece.replace(',', '')))
        except ValueError:
            continue
    return found
def is_verified(text, calls):
    results = [call.value for call in calls if call.ok and isinstance(call.value, (int, float)) and not isinstance(call.value, bool)]
    if not results:
        return False
    last = results[-1]
    return any(math.isclose(last, number, rel_tol=1e-3, abs_tol=1e-9) for number in numbers_in(text))
def run_assistant(question, chat_fn=chat, max_rounds=5):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]
    calls = []
    for round_number in range(1, max_rounds + 1):
        response = chat_fn(messages=messages, tools=registry.schemas())
        message = response.message
        messages.append(message)
        if not message.tool_calls:
            text = message.content or ''
            return Answer(text, is_verified(text, calls), calls, round_number)
        for tool_call in message.tool_calls:
            result = registry.execute(tool_call.function.name, **tool_call.function.arguments)
            calls.append(result)
            messages.append({'role': 'tool', 'tool_name': result.name, 'content': result.content})
    return Answer('I could not finish inside the round limit.', False, calls, max_rounds)
