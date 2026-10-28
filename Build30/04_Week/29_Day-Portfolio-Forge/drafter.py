from typing import Callable
from config import MAX_RETRIES
from receipts import failures, verify
from schemas import BuildSignals, CaseStudy, DraftResult, Receipt
SYSTEM = '''You write short technical case studies for a hiring manager. Use plain words and short sentences. Use only the numbers listed in RECEIPTS. Never write a number that is not in RECEIPTS. Write percentages with the percent sign, never as a fraction like 71/100. Say what you chose and what you rejected. Do not use the words leverage, robust, seamless, comprehensive or innovative.'''
def study_text(study: CaseStudy) -> str:
    return ' '.join([study.headline, study.problem, study.decision, study.tradeoff, study.result, study.what_broke])
def receipt_lines(receipts: list[Receipt]) -> str:
    return '\n'.join(f'- {r.label}: {r.value:g} {r.unit}'.rstrip() for r in receipts)
def build_prompt(build: BuildSignals, role: str, receipts: list[Receipt], decisions: list[str]) -> str:
    return f'''READER: a hiring manager for this role: {role}
BUILD: day {build.day}, {build.title}
README EXCERPT: {build.summary}
DECISION COMMITS: {'; '.join(decisions) or 'none found'}
RECEIPTS:
{receipt_lines(receipts)}
Write the case study as JSON. headline: 12 words or fewer. problem: 2 sentences. decision: 2 sentences. tradeoff: 1 or 2 sentences about what the choice cost. result: use the receipts. what_broke: what broke and how it was fixed.'''
def fallback_study(build: BuildSignals, decisions: list[str]) -> CaseStudy:
    decision = decisions[0] if decisions else '[fill in: the decision you made]'
    return CaseStudy(
        headline=f'{build.title}: what I decided and what it cost',
        problem=f'Day {build.day} of Build30 asked me to build {build.title.lower()}. [fill in: the problem in your own words]',
        decision=f'This is the commit that shows the call: {decision}.',
        tradeoff='[fill in: what this choice cost you]',
        result=f'It took {build.commits} commits across {build.sessions} work sessions. [fill in: your measured result]',
        what_broke='[fill in: what broke and how you fixed it]',
    )
def draft(
    build: BuildSignals,
    role: str,
    receipts: list[Receipt],
    decisions: list[str],
    llm: Callable[[str, type[CaseStudy], str], CaseStudy],
    retries: int = MAX_RETRIES,
) -> DraftResult:
    prompt = build_prompt(build, role, receipts, decisions)
    feedback, error = '', ''
    for attempt in range(1, retries + 2):
        try:
            study = llm(prompt + feedback, CaseStudy, SYSTEM)
        except Exception as problem:
            error = f'{type(problem).__name__}: {problem}'
            break
        findings = verify(study_text(study), receipts)
        bad = failures(findings)
        if not bad:
            return DraftResult(study=study, source='model', attempts=attempt, findings=findings)
        error = 'numbers not in receipts: ' + ', '.join(item.raw for item in bad)
        feedback = '\nYour last answer used numbers that are not in RECEIPTS: ' + ', '.join(item.raw for item in bad) + '. Rewrite it without them.'
    return DraftResult(study=fallback_study(build, decisions), source='template', attempts=attempt, findings=[], error=error)
