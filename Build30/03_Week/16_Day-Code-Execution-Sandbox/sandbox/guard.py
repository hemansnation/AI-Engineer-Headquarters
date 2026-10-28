import os
import sys
import traceback
WORKDIR = os.path.realpath(os.getcwd())
READ_ROOTS = tuple({os.path.realpath(p) for p in (WORKDIR, sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)})
READ_FILES = {'/dev/null', '/dev/urandom', '/dev/random'}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
BLOCKED_EVENTS = {
    'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'socket.gethostbyname',
    'subprocess.Popen', 'os.system', 'os.exec', 'os.fork', 'os.forkpty',
    'os.posix_spawn', 'os.spawn', 'os.kill', 'os.killpg', 'os.symlink', 'os.link',
    'ctypes.dlopen', 'ctypes.dlsym',
}
BLOCKED_IMPORTS = {'ctypes', '_ctypes'}
PATH_EVENTS = {'os.remove', 'os.rmdir', 'os.rename', 'os.mkdir', 'os.chmod', 'os.truncate'}
def inside(path, roots):
    try:
        real = os.path.realpath(os.fsdecode(path))
    except Exception:
        return False
    return any(real == root or real.startswith(root + os.sep) for root in roots)
def guard(event, args):
    if event in BLOCKED_EVENTS:
        raise PermissionError('blocked by sandbox: ' + event)
    if event == 'import' and args[0] in BLOCKED_IMPORTS:
        raise ImportError('blocked by sandbox: import ' + args[0])
    if event == 'open':
        path, mode, flags = args
        if isinstance(path, int) or path in READ_FILES:
            return
        wants_write = isinstance(flags, int) and bool(flags & WRITE_FLAGS)
        roots = (WORKDIR,) if wants_write else READ_ROOTS
        if not inside(path, roots):
            raise PermissionError('blocked by sandbox: open ' + str(path))
    if event in PATH_EVENTS:
        for arg in args:
            if isinstance(arg, (str, bytes)) and not inside(arg, (WORKDIR,)):
                raise PermissionError('blocked by sandbox: ' + event)
def main():
    script = sys.argv[1]
    with open(script, encoding='utf-8') as handle:
        source = handle.read()
    code = compile(source, 'main.py', 'exec')
    sys.addaudithook(guard)
    try:
        exec(code, {'__name__': '__main__', '__file__': 'main.py'})
    except SystemExit:
        raise
    except BaseException as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)
        sys.exit(1)
main()
