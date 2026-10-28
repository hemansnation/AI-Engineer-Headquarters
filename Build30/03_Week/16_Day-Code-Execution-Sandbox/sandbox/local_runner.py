import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import psutil
from .result import Limits, SandboxResult
GUARD = Path(__file__).with_name('guard.py')
def _posix_limits(limits):
    def apply():
        import resource
        cpu = int(limits.timeout_s) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        size = limits.max_file_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply
def _memory_mb(proc):
    try:
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            total += child.memory_info().rss
        return total / (1024 * 1024)
    except psutil.Error:
        return 0.0
def kill_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for proc in parent.children(recursive=True) + [parent]:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
def read_capped(path, cap):
    data = Path(path).read_bytes()[:cap]
    return data.decode('utf-8', errors='replace')
def watch(process, out_path, err_path, limits, started, check_memory):
    handle = psutil.Process(process.pid)
    while process.poll() is None:
        if time.monotonic() - started > limits.timeout_s:
            return 'time'
        if check_memory and _memory_mb(handle) > limits.memory_mb:
            return 'memory'
        written = os.path.getsize(out_path) + os.path.getsize(err_path)
        if written > limits.max_output_bytes:
            return 'output'
        time.sleep(0.02)
    written = os.path.getsize(out_path) + os.path.getsize(err_path)
    return 'output' if written > limits.max_output_bytes else ''
def run_local(code: str, limits: Limits = Limits()) -> SandboxResult:
    work = Path(tempfile.mkdtemp(prefix='sbx_work_'))
    io_dir = Path(tempfile.mkdtemp(prefix='sbx_io_'))
    started = time.monotonic()
    try:
        (work / 'main.py').write_text(code, encoding='utf-8')
        out_path = io_dir / 'stdout.txt'
        err_path = io_dir / 'stderr.txt'
        env = {'PATH': '/usr/bin:/bin'}
        if os.name == 'nt':
            env['SYSTEMROOT'] = os.environ.get('SYSTEMROOT', '')
        command = [sys.executable, '-I', '-S', '-B', '-X', 'utf8', str(GUARD), str(work / 'main.py')]
        with open(out_path, 'wb') as out, open(err_path, 'wb') as err:
            process = subprocess.Popen(
                command, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                preexec_fn=_posix_limits(limits) if os.name == 'posix' else None,
            )
            killed = watch(process, out_path, err_path, limits, started, True)
            if killed:
                kill_tree(process.pid)
            process.wait()
        stdout = read_capped(out_path, limits.max_output_bytes)
        stderr = read_capped(err_path, limits.max_output_bytes).replace(str(work), '/sandbox')
        code_out = process.returncode
        return SandboxResult(
            ok=(code_out == 0 and not killed), stdout=stdout, stderr=stderr, exit_code=code_out,
            killed_reason=killed, duration_s=round(time.monotonic() - started, 3), backend='local',
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(io_dir, ignore_errors=True)
