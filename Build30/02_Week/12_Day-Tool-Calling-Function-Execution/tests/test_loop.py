from ollama import ChatResponse, Message
from loop import is_verified, numbers_in, run_assistant
from registry import ToolResult
def reply(content=None, calls=None):
    tool_calls = [
        Message.ToolCall(function=Message.ToolCall.Function(name=name, arguments=arguments))
        for name, arguments in (calls or [])
    ]
    return ChatResponse(message=Message(role='assistant', content=content, tool_calls=tool_calls or None))
def scripted(*replies):
    queue = list(replies)
    seen = []
    def fake_chat(messages, tools):
        seen.append(len(messages))
        return queue.pop(0)
    fake_chat.seen = seen
    return fake_chat
def test_full_round_trip_gives_a_verified_answer():
    fake = scripted(
        reply(calls=[('calculate', {'expression': '(15 / 100) * 2480 + 37 * 12'})]),
        reply(content='15 percent of 2480 plus 37 times 12 is 816.'),
    )
    answer = run_assistant('Do the math', chat_fn=fake)
    assert answer.verified and answer.rounds == 2
    assert answer.calls[0].value == 816
    assert fake.seen == [2, 4]
def test_model_that_ignores_the_tool_result_is_not_verified():
    fake = scripted(
        reply(calls=[('calculate', {'expression': '6 * 7'})]),
        reply(content='The answer is 41.'),
    )
    assert not run_assistant('6 times 7', chat_fn=fake).verified
def test_error_goes_back_to_the_model_and_it_recovers():
    fake = scripted(
        reply(calls=[('calculate', {'expression': '10 / 0'})]),
        reply(calls=[('calculate', {'expression': '10 / 4'})]),
        reply(content='10 divided by 4 is 2.5.'),
    )
    answer = run_assistant('10 over 4', chat_fn=fake)
    assert [call.ok for call in answer.calls] == [False, True]
    assert answer.verified and answer.rounds == 3
def test_unknown_tool_does_not_crash_the_loop():
    fake = scripted(
        reply(calls=[('format_disk', {'drive': 'C'})]),
        reply(content='I cannot do that.'),
    )
    answer = run_assistant('hello', chat_fn=fake)
    assert not answer.calls[0].ok and not answer.verified
def test_two_tool_calls_in_one_turn():
    fake = scripted(
        reply(calls=[('get_stock_price', {'ticker': 'AAPL'}), ('get_stock_price', {'ticker': 'MSFT'})]),
        reply(content='AAPL is 227.52 and MSFT is 415.10.'),
    )
    answer = run_assistant('two prices', chat_fn=fake)
    assert len(answer.calls) == 2 and answer.verified
def test_round_limit_stops_a_looping_model():
    loop_forever = scripted(*[reply(calls=[('calculate', {'expression': '1 + 1'})]) for _ in range(5)])
    answer = run_assistant('loop', chat_fn=loop_forever, max_rounds=5)
    assert answer.rounds == 5 and not answer.verified and len(answer.calls) == 5
def test_number_helpers():
    assert numbers_in('Total: 1,234.50 dollars, then 7.') == [1234.5, 7.0]
    assert not is_verified('no numbers here', [])
    call = ToolResult('calculate', {}, True, 816, '816', 0.0)
    assert is_verified('It is 816.', [call])
