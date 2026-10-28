EXAMPLES = [
    ('Which river runs through the town where the Orlin Bridge stands?', [
        ('I need to find where the Orlin Bridge stands, then find the river of that town.', 'Search[Orlin Bridge]', 'The Orlin Bridge is a stone arch bridge in the town of Pell. It was finished in 1890.'),
        ('The bridge is in Pell. Now I need the river that runs through Pell.', 'Search[Pell]', 'Pell is a market town on the Ashe river. It has a mill. It has a market. It has a church. It has a school.'),
        ('The river is the Ashe river, so the answer is Ashe.', 'Finish[Ashe]', ''),
    ]),
    ('How many people live in the town where the Orlin Bridge stands?', [
        ('I need the town of the Orlin Bridge first.', 'Search[Orlin Bridge]', 'The Orlin Bridge is a stone arch bridge in the town of Pell. It was finished in 1890.'),
        ('The town is Pell. I need its population, so I will search Pell.', 'Search[Pell]', 'Pell is a market town on the Ashe river. It has a mill. It has a market. It has a church. It has a school.'),
        ('The population is not shown. I will look up the keyword population on this page.', 'Lookup[population]', 'It has a population of 5000.'),
        ('The population is 5000.', 'Finish[5000]', ''),
    ]),
]
PAPER_HEADER = (
    'Solve a question answering task with interleaving Thought, Action, Observation steps. '
    'Thought can reason about the current situation, and Action can be three types:\n'
    '(1) Search[entity], which returns the first sentences of the page if it exists, or else suggests similar titles.\n'
    '(2) Lookup[keyword], which returns the next sentence containing the keyword on the last page you searched.\n'
    '(3) Finish[answer], which returns the answer and finishes the task.\n'
    'Here are some examples.\n'
)
def render_examples(numbered):
    blocks = []
    for question, steps in EXAMPLES:
        lines = ['Question: ' + question]
        for number, (thought, action, observation) in enumerate(steps, start=1):
            tag = ' ' + str(number) if numbered else ''
            lines.append('Thought' + tag + ': ' + thought)
            lines.append('Action' + tag + ': ' + action)
            if observation:
                lines.append('Observation' + tag + ': ' + observation)
        blocks.append('\n'.join(lines))
    return '\n'.join(blocks) + '\n'
def paper_prompt_start():
    return PAPER_HEADER + render_examples(True)
def mine_system(registry):
    tools = '\n'.join('- ' + name + '[...]: ' + text for name, text in registry.descriptions.items())
    return (
        'You answer questions by alternating Thought and Action lines. After each Action you will receive an Observation.\n'
        'Write exactly one Thought and one Action per reply. Available actions:\n'
        + tools + '\n- Finish[answer]: give the short final answer and stop.\n'
        'Here are some examples.\n' + render_examples(False)
    )
