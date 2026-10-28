import json
from datetime import datetime, timezone
from guard.config import LOG_FILE
def append_record(record, path=LOG_FILE):
    record = {'time': datetime.now(timezone.utc).isoformat(timespec='seconds'), **record}
    with open(path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(record) + '\n')
