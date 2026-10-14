import re
from .models import Chunk
def nearest_heading(text, pos):
    heading = ''
    for m in re.finditer(r'^#{1,6}\s+(.*)$', text[:pos], flags=re.M):
        heading = m.group(1).strip()
    return heading
def line_end(text, pos):
    j = text.find('\n', pos)
    return len(text) if j == -1 else j
def hard_split(text, start, end, counter, size):
    offs = counter.offsets(text[start:end])
    units = []
    for i in range(0, len(offs), size):
        block = offs[i:i + size]
        units.append((start + block[0][0], start + block[-1][1], len(block)))
    return units
def pack(text, units, source, strategy, counter, size, overlap):
    chunks = []
    window = []
    used = 0
    def emit():
        s = window[0][0]
        e = window[-1][1]
        raw = text[s:e]
        s += len(raw) - len(raw.lstrip())
        e -= len(raw) - len(raw.rstrip())
        piece = text[s:e]
        chunks.append(Chunk(
            text=piece,
            source=source,
            index=len(chunks),
            start_char=s,
            end_char=e,
            token_count=counter.count(piece),
            strategy=strategy,
            heading=nearest_heading(text, line_end(text, s)),
        ))
    for unit in units:
        n = unit[2]
        if window and used + n > size:
            emit()
            carry = []
            total = 0
            for u in reversed(window):
                if total + u[2] > overlap:
                    break
                carry.insert(0, u)
                total += u[2]
            window = carry
            used = total
            while window and used + n > size:
                gone = window.pop(0)
                used -= gone[2]
        window.append(unit)
        used += n
    if window:
        emit()
    return chunks
