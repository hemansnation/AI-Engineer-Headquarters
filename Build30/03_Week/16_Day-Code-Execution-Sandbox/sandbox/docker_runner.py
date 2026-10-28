import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from .local_runner import read_capped, watch
from .result import Limits, SandboxResult
IMAGE = 'python:3.13-slim'
BASE = Path('.sandbox_work').resolve()
def docker_ready() -> bool:
    if not shutil.which('docker'):
        return False
    try:
        probe = subprocess.run(['docker', 'info'], capture_output=True, timeout=10)
    except subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0
def run_docker(code: str, limits: Limits = Limits()) -> SandboxResult:
    BASE.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='run_', dir=BASE))
    io_dir = Path(tempfile.mkdtemp(prefix='io_', dir=BASE))
    name = 'sbx-' + uuid.uuid4().hex[:12]
    started = time.monotonic()
    try:
        work.chmod(0o755)
        script = work / 'main.py'
        script.write_text(code, encoding='utf-8')
        script.chmod(0o644)
        out_path = io_dir / 'stdout.txt'
        err_path = io_dir / 'stderr.txt'
        memory = str(limits.memory_mb) + 'm'
        command = [
            'docker', 'run', '--rm', '--name', name,
            '--network', 'none', '--read-only',
            '--tmpfs', '/tmp:rw,size=16m,noexec,nosuid',
            '--memory', memory, '--memory-swap', memory, '--cpus', '0.5', '--pids-limit', '32',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--user', '65534:65534',
            '-v', str(work) + ':/work:ro', '-w', '/work',
            IMAGE, 'python', '-I', '-B', '-X', 'utf8', '/work/main.py',
        ]
        with open(out_path, 'wb') as out, open(err_path, 'wb') as err:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            killed = watch(process, out_path, err_path, limits, started, False)
            if killed:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
            process.wait()
        stdout = read_capped(out_path, limits.max_output_bytes)
        stderr = read_capped(err_path, limits.max_output_bytes)
        exit_code = process.returncode
        if exit_code == 137 and not killed:
            killed = 'memory'
        return SandboxResult(
            ok=(exit_code == 0 and not killed), stdout=stdout, stderr=stderr, exit_code=exit_code,
            killed_reason=killed, duration_s=round(time.monotonic() - started, 3), backend='docker',
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(io_dir, ignore_errors=True)
