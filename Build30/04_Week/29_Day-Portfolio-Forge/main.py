import argparse
import json
import sys
from pathlib import Path
import config
from assembler import render_case_study, render_portfolio
from drafter import draft, fallback_study
from gitlog import commits_by_day, decision_subjects, read_commits
from ranker import rank_all
from receipts import build_receipts
from sessions import active_dates, longest_streak, make_sessions
from signals import scan_all
from skim import skim
from snippet import pick_snippet
from tracker import running_totals
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Portfolio Forge: pick your lead build and write the case study')
    parser.add_argument('--repo', default=str(config.REPO_PATH))
    parser.add_argument('--role', default=config.TARGET_ROLE)
    parser.add_argument('--author', default=config.AUTHOR_NAME)
    parser.add_argument('--lead', type=int, default=None)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--out', default=str(config.OUT_DIR))
    return parser.parse_args(argv)
def run(args: argparse.Namespace, llm=None, fit_fn=None) -> dict:
    repo, out = Path(args.repo).expanduser(), Path(args.out)
    demo = (repo / 'DEMO_DATA.txt').exists()
    commits = read_commits(repo)
    grouped = commits_by_day(commits)
    builds = scan_all(repo, grouped)
    if not builds:
        raise SystemExit('No folders named like day-01 found in ' + str(repo))
    fits = None
    if not args.offline:
        if fit_fn is None:
            from embed import role_fit as fit_fn
        fits = fit_fn(args.role, [f'{b.title}. {b.summary}' for b in builds])
    ranked = rank_all(builds, fits)
    lead = next((item for item in ranked if item.day == args.lead), ranked[0])
    build = next(item for item in builds if item.day == lead.day)
    day_commits = grouped.get(build.day, [])
    receipts = build_receipts(build, day_commits)
    decisions = decision_subjects(day_commits, build.day)
    if args.offline:
        result = None
        study = fallback_study(build, decisions)
        source = 'template (offline)'
    else:
        if llm is None:
            from llm import ask_structured as llm
        result = draft(build, args.role, receipts, decisions, llm)
        study, source = result.study, f'{result.source}, {result.attempts} attempt(s)'
    snippet = pick_snippet(repo, day_commits, build.day)
    case_study = render_case_study(study, args.author, build.day, receipts, snippet, demo)
    report = skim(case_study)
    stats = {
        'builds found': len(builds),
        'commits': len(commits),
        'active days': len(active_dates(commits)),
        'longest streak in days': longest_streak(commits),
        'work sessions': len(make_sessions(commits, config.SESSION_GAP_MINUTES)),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / 'CASE_STUDY.md').write_text(case_study, encoding='utf-8')
    (out / 'PORTFOLIO_README.md').write_text(render_portfolio(args.author, args.role, ranked, lead, stats, demo), encoding='utf-8')
    (out / 'score_report.json').write_text(json.dumps([item.model_dump() for item in ranked], indent=2), encoding='utf-8')
    return {'lead': lead, 'source': source, 'skim': report, 'out': out, 'error': result.error if result else ''}
def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = run(args)
    lead, report, source, error, out = summary['lead'], summary['skim'], summary['source'], summary['error'], summary['out']
    print(f'Lead build: day {lead.day}, {lead.title} (score {lead.score})')
    print(f'Drafted by: {source}')
    if error:
        print(f'Why not the model: {error}')
    print(f'Skim time: {report.seconds} seconds, {report.words} words')
    for issue in report.issues:
        print(f'  to fix: {issue}')
    print(f'Files written to {out}')
    print(f'Token totals: {running_totals()}')
if __name__ == '__main__':
    main()
