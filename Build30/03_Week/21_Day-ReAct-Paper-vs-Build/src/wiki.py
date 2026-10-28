from pathlib import Path
from embeddings import TitleIndex
DATA = Path(__file__).resolve().parent.parent / 'data'
class Wiki:
    def __init__(self, path=DATA / 'mini_wiki.txt'):
        self.pages = {}
        for line in path.read_text(encoding='utf-8').splitlines():
            if ' | ' not in line:
                continue
            title, body = line.split(' | ', 1)
            sentences = [s.strip().rstrip('.') + '.' for s in body.split('. ') if s.strip()]
            self.pages[title.strip()] = sentences
        self.lower = {title.lower(): title for title in self.pages}
        self.index = TitleIndex(list(self.pages))
        self.reset()
    def reset(self):
        self.last_title = None
        self.cursor = 0
    def search(self, entity):
        title = self.lower.get(entity.strip().lower())
        if title:
            self.last_title = title
            self.cursor = 0
            return ' '.join(self.pages[title][:5])
        close = self.index.closest(entity, k=5)
        return 'Could not find ' + entity + '. Similar: ' + ', '.join(close)
    def lookup(self, keyword):
        if self.last_title is None:
            return 'No page is open. Use Search first.'
        sentences = self.pages[self.last_title]
        for i in range(self.cursor, len(sentences)):
            if keyword.lower() in sentences[i].lower():
                self.cursor = i + 1
                return sentences[i]
        return 'No more results for ' + keyword
