import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from chunker import load_chunks
from embedder import embed
from git_history import read_commits, summarize
from llm import ask, LLMUnavailable
from store import EvidenceStore
from writer import gather, draft_case_study, offline_draft, short_posts
load_dotenv()
NOTES = Path('data/build_notes')
INDEX = Path('data/index')
OUT = Path('out')
def index_notes():
    chunks = load_chunks(NOTES)
    store = EvidenceStore(embed)
    store.add_many(chunks)
    store.save(INDEX)
    print(f'Indexed {len(chunks)} chunks from {NOTES}')
def draft(build):
    OUT.mkdir(exist_ok=True)
    commits = read_commits(os.getenv('REPO_PATH', '.'))
    stats = summarize(commits)
    (OUT / 'build_history.json').write_text(json.dumps(commits, indent=2), encoding='utf-8')
    store = EvidenceStore.load(INDEX, embed)
    evidence = gather(store, build)
    flagged = []
    try:
        case_study, flagged = draft_case_study(build, evidence, stats, ask)
        linkedin, thread = short_posts(case_study, ask)
        (OUT / 'linkedin_post.md').write_text(linkedin, encoding='utf-8')
        (OUT / 'x_thread.md').write_text(thread, encoding='utf-8')
    except LLMUnavailable as error:
        print(f'No local model reachable, writing the offline draft. Reason: {error}')
        case_study = offline_draft(build, evidence)
    (OUT / 'case_study.md').write_text(case_study, encoding='utf-8')
    showcase = {
        'build': build,
        'stats': stats,
        'github': os.getenv('GITHUB_URL', ''),
        'substack': os.getenv('SUBSTACK_URL', ''),
        'linkedin': os.getenv('LINKEDIN_URL', ''),
        'x': os.getenv('X_URL', ''),
        'numbers_still_unverified': flagged,
    }
    (OUT / 'showcase.json').write_text(json.dumps(showcase, indent=2), encoding='utf-8')
    print(f'Done. Stats: {stats}')
    print(f'Numbers still unverified: {flagged}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['index', 'draft'])
    parser.add_argument('--build', default='day-09')
    args = parser.parse_args()
    if args.command == 'index':
        index_notes()
    else:
        draft(args.build)
