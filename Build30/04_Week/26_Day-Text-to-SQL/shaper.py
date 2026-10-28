from config import ROWS_FOR_MODEL
def to_table(result, limit=ROWS_FOR_MODEL):
    if not result.rows:
        return 'No rows matched this query.'
    lines = [' | '.join(result.columns)]
    for row in result.rows[:limit]:
        lines.append(' | '.join('' if value is None else str(value) for value in row))
    shown = min(len(result.rows), limit)
    note = 'Rows shown: ' + str(shown) + ' of ' + str(len(result.rows)) + ' returned.'
    if result.truncated:
        note += ' The query hit the row cap, so more rows exist in the database.'
    elif shown < len(result.rows):
        note += ' Only the first rows are shown here.'
    return '\n'.join(lines) + '\n' + note
