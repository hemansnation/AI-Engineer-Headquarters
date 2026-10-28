from claims import unsupported, mask, PLACEHOLDER
from git_history import stats_text
SECTIONS = [
    ('Problem', 'what problem did this build solve and for whom'),
    ('Decision', 'why I chose this approach and what I rejected'),
    ('Tradeoff', 'what it cost me, what broke and how I fixed it'),
    ('Result', 'numbers I measured, latency, cost, accuracy, pass rate'),
]
MAX_ROUNDS = 3
def gather(store, build=None, per_section=3):
    return {title: store.query(question, per_section, build) for title, question in SECTIONS}
def evidence_texts(evidence):
    return [hit['text'] for hits in evidence.values() for hit in hits]
def build_prompt(build, evidence, stats):
    blocks = []
    for title, hits in evidence.items():
        for hit in hits:
            source, text = hit['source'], hit['text']
            blocks.append(f'[{title} | {source}] {text}')
    joined = '\n'.join(blocks)
    return f'''You are helping a builder write a technical case study for a hiring manager.
Write only from the evidence below, with the sections Problem, Decision, Tradeoff and Result.
Every number you write must appear in the evidence or in the git stats. If a number is missing, write {PLACEHOLDER} instead of guessing.
Write in first person, plain words, short paragraphs. No code. No links.
Build: {build}
Git stats: {stats_text(stats)}
Evidence:
{joined}'''
def draft_case_study(build, evidence, stats, chat):
    sources = evidence_texts(evidence) + [stats_text(stats)]
    text = chat(build_prompt(build, evidence, stats))
    for _ in range(MAX_ROUNDS):
        bad = unsupported(text, sources)
        if not bad:
            return text, []
        text = chat(f'These numbers are not in the evidence: {bad}. Rewrite the case study and remove them or use {PLACEHOLDER}.\n\n{text}')
    bad = unsupported(text, sources)
    return mask(text, bad), bad
def offline_draft(build, evidence):
    parts = [f'# {build} case study', 'This draft was built without a model. Edit every section by hand.']
    for title, hits in evidence.items():
        parts.append(f'## {title}')
        parts.append(' '.join(hit['text'] for hit in hits[:2]))
    return '\n\n'.join(parts)
def short_posts(case_study, chat):
    linkedin = chat(f'Write a LinkedIn post of 1000 to 1800 characters from this case study. Plain words, one idea, no links, no code, no new numbers.\n\n{case_study}')
    thread = chat(f'Write an X thread of 5 posts from this case study, each under 280 characters, separated by a line with three dashes. No new numbers.\n\n{case_study}')
    return mask(linkedin, unsupported(linkedin, [case_study])), mask(thread, unsupported(thread, [case_study]))
