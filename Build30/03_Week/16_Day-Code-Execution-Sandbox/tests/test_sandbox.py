import pytest
from break_it import ATTACKS, PWN_FILE
from sandbox import Limits, run_python
LIMITS = Limits(timeout_s=4.0, memory_mb=100, max_output_bytes=5000)
@pytest.mark.parametrize('name,code,check,backends', ATTACKS)
def test_attack_is_stopped(name, code, check, backends, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-key-for-the-test')
    PWN_FILE.unlink(missing_ok=True)
    result = run_python(code, LIMITS)
    if result.backend not in backends:
        pytest.skip('not meant for ' + result.backend)
    assert check(result), result.for_model()
