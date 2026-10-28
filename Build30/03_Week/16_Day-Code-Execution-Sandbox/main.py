import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from break_it import run_all
from sandbox import run_python
def main():
    command = sys.argv[1] if len(sys.argv) > 1 else 'help'
    if command == 'break':
        sys.exit(0 if run_all() else 1)
    elif command == 'run':
        code = Path(sys.argv[2]).read_text(encoding='utf-8')
        print(run_python(code).for_model())
    elif command == 'agent':
        from agent import solve
        task = ' '.join(sys.argv[2:]) or 'Print the first 10 square numbers.'
        solve(task)
    else:
        print('use: python main.py break | run file.py | agent your task here')
main()
