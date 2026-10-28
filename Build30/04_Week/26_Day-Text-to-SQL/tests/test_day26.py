import sqlite3
import pytest
from guard import SQLRejected, validate
from db import QueryResult, run_query
from shaper import to_table
from data.build_db import build
@pytest.fixture(scope='module')
def db_path(tmp_path_factory):
    return str(build(tmp_path_factory.mktemp('db') / 'lakeview.db'))
@pytest.mark.parametrize('sql', ['DROP TABLE orders', 'SELECT id FROM orders; DELETE FROM orders', 'SELECT * FROM orders', 'SELECT email FROM customers', 'SELECT staff_name FROM staff_salaries', 'PRAGMA query_only = OFF', 'SELECT load_extension(1)'])
def test_validator_blocks_attacks(sql):
    with pytest.raises(SQLRejected):
        validate(sql)
def test_validator_adds_a_limit():
    assert validate('SELECT id FROM orders', max_rows=50).endswith('LIMIT 50')
def test_validator_lowers_a_big_limit():
    assert validate('SELECT id FROM orders LIMIT 100000', max_rows=50).endswith('LIMIT 50')
def test_validator_allows_join_count_and_cte():
    validate('SELECT c.city, COUNT(*) AS n FROM orders o JOIN customers c ON c.id = o.customer_id GROUP BY c.city')
    validate('WITH recent AS (SELECT id FROM orders WHERE order_date >= ' + chr(39) + '2026-09-01' + chr(39) + ') SELECT COUNT(*) FROM recent')
def test_string_that_looks_like_sql_is_fine():
    validate('SELECT id FROM warranty_claims WHERE issue = ' + chr(39) + 'drop table' + chr(39))
def test_connection_refuses_writes_even_without_validator(db_path):
    with pytest.raises(sqlite3.DatabaseError):
        run_query('DELETE FROM orders', path=db_path)
    assert run_query('SELECT COUNT(*) FROM orders', path=db_path).rows[0][0] == 1500
def test_raw_read_only_file_refuses_writes(db_path):
    con = sqlite3.connect('file:' + db_path + '?mode=ro', uri=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute('DELETE FROM orders')
    con.close()
def test_connection_refuses_hidden_column(db_path):
    with pytest.raises(sqlite3.DatabaseError):
        run_query('SELECT email FROM customers', path=db_path)
def test_slow_query_is_stopped(db_path):
    slow = 'WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n'
    with pytest.raises(TimeoutError):
        run_query(slow, path=db_path, timeout=0.2)
def test_row_cap_marks_truncation(db_path):
    result = run_query('SELECT id FROM orders', path=db_path, max_rows=10)
    assert len(result.rows) == 10 and result.truncated
def test_empty_and_truncated_text():
    assert 'No rows' in to_table(QueryResult(['id'], [], False, 1))
    assert 'row cap' in to_table(QueryResult(['id'], [(1,), (2,)], True, 1))
