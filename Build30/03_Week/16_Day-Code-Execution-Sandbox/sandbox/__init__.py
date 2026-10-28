import os
from .docker_runner import docker_ready, run_docker
from .local_runner import run_local
from .result import Limits, SandboxResult, clip
def run_python(code: str, limits: Limits = None, backend: str = None) -> SandboxResult:
    limits = limits or Limits()
    backend = backend or os.getenv('SANDBOX_BACKEND', 'auto')
    if backend == 'docker' or (backend == 'auto' and docker_ready()):
        return run_docker(code, limits)
    return run_local(code, limits)
