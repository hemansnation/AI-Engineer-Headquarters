import json
import threading
from pathlib import Path
_lock = threading.Lock()
def append_usage(path: str, record: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + '\n'
    with _lock:
        with file_path.open('a', encoding='utf-8') as file:
            file.write(line)
