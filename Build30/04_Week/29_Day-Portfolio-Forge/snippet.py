import ast
from collections import Counter
from pathlib import Path
from pydantic import BaseModel
from gitlog import day_of
from schemas import Commit
SKIP_NAMES = {'main', '__init__', 'wrapper'}
BRANCHES = (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.IfExp, ast.comprehension)
class Snippet(BaseModel):
    path: str
    name: str
    code: str
def complexity(node: ast.AST) -> int:
    return 1 + sum(isinstance(child, BRANCHES) for child in ast.walk(node))
def best_function(source: str, max_lines: int = 15) -> tuple[str, str] | None:
    lines = source.splitlines()
    best = None
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name in SKIP_NAMES:
            continue
        length = node.end_lineno - node.lineno + 1
        if length < 3 or length > max_lines:
            continue
        score = complexity(node)
        if best is None or score > best[0]:
            best = (score, node.name, '\n'.join(lines[node.lineno - 1:node.end_lineno]))
    return (best[1], best[2]) if best else None
def pick_snippet(repo: Path, commits: list[Commit], day: int) -> Snippet | None:
    churn = Counter()
    for commit in commits:
        for change in commit.files:
            if change.path.endswith('.py') and day_of(change.path) == day:
                churn[change.path] += change.added + change.removed
    for path, _ in churn.most_common():
        file = repo / path
        if not file.exists():
            continue
        try:
            found = best_function(file.read_text(encoding='utf-8'))
        except SyntaxError:
            continue
        if found:
            return Snippet(path=path, name=found[0], code=found[1])
    return None
