from git_history import longest_streak, summarize
def make(*days):
    return [{'sha': 'abc1234', 'date': day, 'subject': 'work'} for day in days]
def test_three_days_in_a_row():
    assert longest_streak(make('2026-09-25', '2026-09-26', '2026-09-27')) == 3
def test_gap_breaks_the_streak():
    assert longest_streak(make('2026-09-25', '2026-09-26', '2026-09-28', '2026-09-29', '2026-09-30')) == 3
def test_month_boundary_counts():
    assert longest_streak(make('2026-09-30', '2026-10-01')) == 2
def test_duplicate_days_count_once():
    assert summarize(make('2026-09-25', '2026-09-25'))['active_days'] == 1
