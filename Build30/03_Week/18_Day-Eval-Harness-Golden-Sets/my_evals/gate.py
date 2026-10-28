from .config import MAX_DROP
from .report import compare_runs
def check_gate(base, new, max_drop=MAX_DROP):
    diff = compare_runs(base, new)
    reasons = []
    if diff['broke_must_pass']:
        reasons.append('must-pass cases broke: ' + str(diff['broke_must_pass']))
    if diff['delta'] < -max_drop:
        drop, limit = abs(diff['delta']) * 100, max_drop * 100
        reasons.append(f'pass rate dropped {drop:.1f} points, limit is {limit:.1f}')
    return len(reasons) == 0, reasons
