import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from config import MAX_ROWS
from schema_cards import ALLOWED_TABLES, HIDDEN_COLUMNS
BLOCKED_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Merge, exp.TruncateTable, exp.Command, exp.Pragma, exp.Attach, exp.Detach, exp.Transaction, exp.Commit, exp.Set, exp.Into)
BLOCKED_FUNCTIONS = {'load_extension', 'readfile', 'writefile', 'edit', 'fts3_tokenizer', 'randomblob', 'zeroblob'}
class SQLRejected(Exception):
    pass
def parse_single(sql):
    try:
        trees = [tree for tree in sqlglot.parse(sql, read='sqlite') if tree is not None]
    except ParseError as error:
        raise SQLRejected('SQL does not parse: ' + str(error).splitlines()[0])
    if len(trees) != 1:
        raise SQLRejected('Exactly one statement is allowed, found ' + str(len(trees)) + '.')
    return trees[0]
def check_tree(tree):
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SQLRejected('Only SELECT queries are allowed, found ' + type(tree).__name__ + '.')
    for node in tree.walk():
        if isinstance(node, BLOCKED_NODES):
            raise SQLRejected(type(node).__name__ + ' is not allowed.')
        if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
            raise SQLRejected('SELECT * is not allowed. Name the columns you need.')
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in BLOCKED_FUNCTIONS:
                raise SQLRejected('Function ' + name + ' is not allowed.')
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name in cte_names:
            continue
        if table.args.get('db') or name not in ALLOWED_TABLES:
            raise SQLRejected('Table ' + table.sql() + ' is not in the allowed list.')
    for column in tree.find_all(exp.Column):
        if column.name.lower() in HIDDEN_COLUMNS:
            raise SQLRejected('Column ' + column.name + ' holds personal data and cannot be used.')
def cap_rows(tree, max_rows=MAX_ROWS):
    limit = tree.args.get('limit')
    if limit is None:
        return tree.limit(max_rows)
    value = limit.expression
    if not isinstance(value, exp.Literal) or not value.is_int or int(value.this) > max_rows:
        return tree.limit(max_rows)
    return tree
def validate(sql, max_rows=MAX_ROWS):
    tree = parse_single(sql)
    check_tree(tree)
    return cap_rows(tree, max_rows).sql(dialect='sqlite')
