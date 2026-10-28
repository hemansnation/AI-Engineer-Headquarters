from dataclasses import dataclass
@dataclass(frozen=True)
class Limits:
    timeout_s: float = 5.0
    memory_mb: int = 128
    max_output_bytes: int = 20000
    max_file_mb: int = 1
@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    killed_reason: str
    duration_s: float
    backend: str
    def for_model(self) -> str:
        if self.ok:
            return 'STATUS: success\nSTDOUT:\n' + (clip(self.stdout) or '(empty)')
        lines = ['STATUS: failed']
        if self.killed_reason:
            lines.append('KILLED: your code hit the ' + self.killed_reason + ' limit')
        lines.append('EXIT CODE: ' + str(self.exit_code))
        if self.stdout:
            lines.append('STDOUT:\n' + clip(self.stdout))
        if self.stderr:
            lines.append('STDERR:\n' + clip(self.stderr, 1500))
        return '\n'.join(lines)
def clip(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    hidden = len(text) - limit
    return text[:half] + '\n...[' + str(hidden) + ' characters hidden]...\n' + text[-half:]
