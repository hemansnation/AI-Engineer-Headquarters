import os
from pathlib import Path
from sandbox import Limits, run_python
PWN_FILE = Path('/tmp/sbx_pwn.txt')
ATTACKS = [
    ('normal code works', 'print(sum(range(10)))',
     lambda r: r.ok and r.stdout.strip() == '45', ('local', 'docker')),
    ('clean exception', 'raise ValueError(\'boom\')',
     lambda r: (not r.ok) and 'ValueError' in r.stderr and r.exit_code == 1 and not r.killed_reason, ('local', 'docker')),
    ('infinite loop', 'while True:\n    pass',
     lambda r: r.killed_reason == 'time', ('local', 'docker')),
    ('memory hog', 'data = []\nwhile True:\n    data.append(\'x\' * 10_000_000)',
     lambda r: r.killed_reason == 'memory', ('local', 'docker')),
    ('output flood', 'while True:\n    print(\'x\' * 1000)',
     lambda r: r.killed_reason == 'output', ('local', 'docker')),
    ('read system file', 'print(open(\'/etc/passwd\').read())',
     lambda r: (not r.ok) and 'root:' not in r.stdout, ('local',)),
    ('leak env secret', 'import os\nprint(os.environ.get(\'GEMINI_API_KEY\', \'missing\'))',
     lambda r: r.ok and r.stdout.strip() == 'missing', ('local', 'docker')),
    ('call the network', 'import socket\nsocket.create_connection((\'example.com\', 80), timeout=2)',
     lambda r: not r.ok, ('local', 'docker')),
    ('write outside folder', 'open(\'/tmp/sbx_pwn.txt\', \'w\').write(\'x\')',
     lambda r: not PWN_FILE.exists(), ('local', 'docker')),
    ('start a process', 'import os\nos.system(\'echo pwned\')',
     lambda r: (not r.ok) and 'pwned' not in r.stdout, ('local',)),
    ('load ctypes', 'import ctypes',
     lambda r: not r.ok, ('local',)),
]
def run_all(backend: str = None) -> bool:
    os.environ['GEMINI_API_KEY'] = 'fake-key-for-the-test'
    limits = Limits(timeout_s=4.0, memory_mb=100, max_output_bytes=5000)
    passed = 0
    total = 0
    for name, code, check, backends in ATTACKS:
        PWN_FILE.unlink(missing_ok=True)
        result = run_python(code, limits, backend)
        if result.backend not in backends:
            continue
        total += 1
        good = bool(check(result))
        passed += good
        print(('PASS  ' if good else 'FAIL  ') + name.ljust(24) + result.backend.ljust(8) + str(result.duration_s) + 's')
    print(str(passed) + ' of ' + str(total) + ' attacks stopped')
    return passed == total
if __name__ == '__main__':
    raise SystemExit(0 if run_all() else 1)
