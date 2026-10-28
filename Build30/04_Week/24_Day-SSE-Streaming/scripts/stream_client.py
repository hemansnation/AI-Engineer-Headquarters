import argparse
import sys
import time
import httpx
def parse_frame(frame):
    event = 'message'
    data = []
    for line in frame.split('\n'):
        if line.startswith(':') or ':' not in line:
            continue
        field, value = line.split(':', 1)
        if value.startswith(' '):
            value = value[1:]
        if field == 'event':
            event = value
        if field == 'data':
            data.append(value)
    return event, '\n'.join(data)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', nargs='*')
    parser.add_argument('--url', default='http://localhost:8000/stream')
    parser.add_argument('--leave-after', type=int, default=0)
    parser.add_argument('--raw', action='store_true')
    args = parser.parse_args()
    prompt = ' '.join(args.prompt) or 'Explain Server-Sent Events in one paragraph.'
    started = time.perf_counter()
    first = None
    tokens = 0
    with httpx.stream('POST', args.url, json={'prompt': prompt}, timeout=None) as response:
        response.raise_for_status()
        if args.raw:
            for line in response.iter_lines():
                print(repr(line))
            return
        buffer = ''
        for piece in response.iter_text():
            buffer += piece
            while '\n\n' in buffer:
                frame, buffer = buffer.split('\n\n', 1)
                event, data = parse_frame(frame)
                if event == 'token':
                    if first is None:
                        first = time.perf_counter() - started
                    tokens += 1
                    print(data, end='', flush=True)
                    if args.leave_after and tokens >= args.leave_after:
                        print('\n[leaving early]')
                        return
                if event == 'error':
                    print('\n[' + data + ']')
    total = time.perf_counter() - started
    print(f'\nfirst token: {first * 1000:.0f} ms, tokens: {tokens}, total: {total * 1000:.0f} ms', file=sys.stderr)
if __name__ == '__main__':
    main()
