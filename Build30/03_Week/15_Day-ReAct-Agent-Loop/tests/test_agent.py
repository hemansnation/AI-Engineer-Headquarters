import numpy as np
from agent import tools
from agent.loop import run_agent
from agent.parser import parse
def scripted(replies):
    state = {'index': 0}
    def fake_llm(messages, stop=None):
        reply = replies[min(state['index'], len(replies) - 1)]
        state['index'] += 1
        size = sum(len(message['content']) for message in messages) // 4
        return {'text': reply, 'prompt_tokens': size, 'completion_tokens': len(reply) // 4}
    return fake_llm
def test_parser_reads_action():
    result = parse('Thought: I need math\nAction: calculator\nAction Input: 3 * 12')
    assert result['kind'] == 'action'
    assert result['tool'] == 'calculator'
    assert result['input'] == '3 * 12'
def test_parser_reads_final_and_ignores_think_block():
    result = parse('<think>hmm</think>Thought: done\nFinal Answer: 36')
    assert result['kind'] == 'final'
    assert result['answer'] == '36'
def test_parser_strips_quotes_around_input():
    quoted = chr(39) + 'strawberry,r' + chr(39)
    result = parse('Action: count_letter\nAction Input: ' + quoted)
    assert result['input'] == 'strawberry,r'
def test_parser_flags_invalid_text():
    assert parse('I think the answer is 36')['kind'] == 'invalid'
def test_agent_finishes_with_two_tools():
    llm = scripted([
        'Thought: count first\nAction: count_letter\nAction Input: strawberry,r',
        'Thought: now multiply\nAction: calculator\nAction Input: 3*12',
        'Thought: I know the answer\nFinal Answer: 36',
    ])
    result = run_agent('r in strawberry times 12', llm=llm)
    assert result['status'] == 'finished'
    assert result['steps'] == 3
    assert result['trace'][0]['observation'] == '3'
    assert result['trace'][1]['observation'] == '36'
    assert result['answer'] == '36'
def test_prompt_tokens_grow_every_step():
    llm = scripted([
        'Action: calculator\nAction Input: 1+1',
        'Action: calculator\nAction Input: 2+2',
        'Final Answer: done',
    ])
    rows = run_agent('x', llm=llm)['tokens'].rows
    assert rows[0]['prompt'] < rows[1]['prompt'] < rows[2]['prompt']
def test_cap_stops_the_loop_and_forces_an_answer():
    counter = {'n': 0}
    def endless(messages, stop=None):
        counter['n'] += 1
        if 'out of steps' in messages[-1]['content']:
            text = 'Final Answer: best guess'
        else:
            text = 'Action: calculator\nAction Input: ' + str(counter['n']) + '+1'
        return {'text': text, 'prompt_tokens': 10, 'completion_tokens': 5}
    result = run_agent('x', llm=endless, max_iterations=3)
    assert result['status'] == 'hit_cap'
    assert result['answer'] == 'best guess'
    assert counter['n'] == 4
def test_guard_blocks_repeated_calls():
    llm = scripted(['Action: calculator\nAction Input: 1+1'] * 4 + ['Final Answer: stuck'])
    result = run_agent('x', llm=llm, max_iterations=5)
    texts = [item['observation'] for item in result['trace'] if item['observation']]
    assert any('already called' in text for text in texts)
def test_unknown_tool_becomes_an_observation():
    llm = scripted(['Action: weather\nAction Input: Bhopal', 'Final Answer: no tool'])
    result = run_agent('x', llm=llm)
    assert result['trace'][0]['observation'].startswith('Error: no tool called weather')
def test_bad_tool_input_does_not_crash_the_loop():
    llm = scripted(['Action: count_letter\nAction Input: strawberry', 'Final Answer: ok'])
    result = run_agent('x', llm=llm)
    assert result['trace'][0]['observation'].startswith('Error: ValueError')
def test_calculator_blocks_code():
    try:
        tools.calculator('__import__(1)')
    except ValueError:
        return
    raise AssertionError('calculator ran something it should not')
class FakeEmbedder:
    words = ['token', 'cost', 'embedding', 'agent', 'loop', 'cap']
    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for text in texts:
            row = np.array([float(word in text.lower()) for word in self.words]) + 0.01
            rows.append(row / np.linalg.norm(row))
        return np.array(rows)
def test_search_notes_returns_the_closest_line(tmp_path, monkeypatch):
    (tmp_path / 'a.txt').write_text('Token cost grows each turn\nAn embedding is a list of numbers\n')
    monkeypatch.setattr(tools, 'NOTES_DIR', tmp_path)
    monkeypatch.setattr(tools, '_cache', {'model': FakeEmbedder()})
    result = tools.search_notes('token cost')
    assert result.splitlines()[0] == '[a.txt] Token cost grows each turn'
