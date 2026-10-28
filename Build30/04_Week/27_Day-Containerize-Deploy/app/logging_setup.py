import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            'time': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created)),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        extra = getattr(record, 'fields', None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload['error'] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level='INFO'):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
