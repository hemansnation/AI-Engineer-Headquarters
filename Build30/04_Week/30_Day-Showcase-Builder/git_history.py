import subprocess
from datetime import date
def read_commits(repo_path):
    result = subprocess.run(['git', '-C', repo_path, 'log', '--date=short', '--pretty=format:%h|%ad|%s'], capture_output=True, text=True, check=True)
    commits = []
    for line in result.stdout.splitlines():
        sha, day, subject = line.split('|', 2)
        commits.append({'sha': sha, 'date': day, 'subject': subject})
    return commits
def longest_streak(commits):
    days = {date.fromisoformat(c['date']).toordinal() for c in commits}
    best = 0
    for day in days:
        if day - 1 not in days:
            run = 1
            while day + run in days:
                run += 1
            best = max(best, run)
    return best
def summarize(commits):
    if not commits:
        raise ValueError('No commits found. Check the repo path in your .env file.')
    dates = sorted(c['date'] for c in commits)
    return {'commits': len(commits), 'first_day': dates[0], 'last_day': dates[-1], 'active_days': len(set(dates)), 'longest_streak': longest_streak(commits)}
def stats_text(stats):
    return ' '.join(f'{key} {value}' for key, value in stats.items())
