import ast
def static_issues(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return [f'Syntax error on line {error.lineno}: {error.msg}']
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return ['No function found. Write the solution as a function.']
    issues = []
    for func in functions:
        args = func.args.posonlyargs + func.args.args + func.args.kwonlyargs
        missing = [a.arg for a in args if a.annotation is None and a.arg not in ('self', 'cls')]
        if missing:
            issues.append(f'Function {func.name}: add type hints for {missing}')
        if func.returns is None:
            issues.append(f'Function {func.name}: add a return type hint')
        if ast.get_docstring(func) is None:
            issues.append(f'Function {func.name}: add a short docstring')
    return issues
