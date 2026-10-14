import tools
from registry import registry
ATTACKS = [
    ('calculate', {'expression': '2 + 2'}),
    ('calculate', {'expression': '__import__(\'os\').system(\'echo hacked\')'}),
    ('calculate', {'expression': '9 ** 9 ** 9'}),
    ('calculate', {'expression': '1 / 0'}),
    ('calculate', {'expression': 5}),
    ('calculate', {'expression': '1 + 1', 'run_as_admin': True}),
    ('get_stock_price', {'ticker': 'AAPL; DROP TABLE users'}),
    ('get_stock_price', {'ticker': 'ZZZZ'}),
    ('delete_all_files', {'path': '/'}),
    ('calculat', {'expression': '1 + 1'}),
]
for name, arguments in ATTACKS:
    shown = ', '.join(f'{key}={value}' for key, value in arguments.items())
    result = registry.execute(name, **arguments)
    print(f'{name}({shown})')
    print(f'    ok={result.ok}  {result.content}')
