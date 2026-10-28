import argparse
import sys
from my_evals.config import GOLDEN, RUNS, JUDGE_MODEL
from my_evals.golden import load_golden
from my_evals.scorers import SCORERS
from my_evals.judge import make_judge
from my_evals import embedder
from my_evals.runner import run_eval, save_run, load_run
from my_evals.report import format_run, format_compare, compare_runs
from my_evals.gate import check_gate
from pipeline.target import make_system
def cmd_run(args):
    cases = load_golden(GOLDEN, SCORERS)
    ctx = {'embed': embedder.similarity, 'judge': make_judge(JUDGE_MODEL)}
    run = run_eval(cases, make_system(args.version), ctx, args.version, GOLDEN)
    save_run(run, RUNS)
    print(format_run(run))
def cmd_compare(args):
    base, new = load_run(RUNS, args.base), load_run(RUNS, args.new)
    print(format_compare(base, new, compare_runs(base, new)))
def cmd_gate(args):
    ok, reasons = check_gate(load_run(RUNS, args.base), load_run(RUNS, args.new))
    print('GATE PASSED' if ok else 'GATE FAILED: ' + '; '.join(reasons))
    sys.exit(0 if ok else 1)
parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='cmd', required=True)
p = sub.add_parser('run'); p.add_argument('version'); p.set_defaults(fn=cmd_run)
p = sub.add_parser('compare'); p.add_argument('base'); p.add_argument('new'); p.set_defaults(fn=cmd_compare)
p = sub.add_parser('gate'); p.add_argument('base'); p.add_argument('new'); p.set_defaults(fn=cmd_gate)
if __name__ == '__main__':
    args = parser.parse_args()
    args.fn(args)
