import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from config import DB_PATH, MAX_ROWS, QUERY_TIMEOUT_SECONDS
from schema_cards import ALLOWED_TABLES, HIDDEN_COLUMNS
ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
@dataclass
class QueryResult:
    columns: list
    rows: list
    truncated: bool
    milliseconds: int
def authorizer(action, arg1, arg2, db_name, source):
    if action not in ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and db_name is not None:
        if arg1 not in ALLOWED_TABLES or arg2 in HIDDEN_COLUMNS:
            return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and arg2 == 'load_extension':
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK
def open_readonly(path=DB_PATH):
    if not Path(path).exists():
        raise FileNotFoundError('No database at ' + str(path) + '. Run python main.py build-db first.')
    con = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
    con.execute('PRAGMA query_only = ON')
    con.set_authorizer(authorizer)
    return con
def run_query(sql, path=DB_PATH, max_rows=MAX_ROWS, timeout=QUERY_TIMEOUT_SECONDS):
    con = open_readonly(path)
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    start = time.monotonic()
    try:
        cursor = con.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [item[0] for item in cursor.description]
    except sqlite3.OperationalError as error:
        if 'interrupted' in str(error):
            raise TimeoutError('Query stopped after ' + format(timeout, 'g') + ' seconds. Ask a narrower question.')
        raise
    finally:
        con.close()
    return QueryResult(columns, rows[:max_rows], len(rows) > max_rows, int((time.monotonic() - start) * 1000))
