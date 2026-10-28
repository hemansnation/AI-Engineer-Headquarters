class Store:
    def __init__(self):
        self.documents = {}
        self.versions = {}
        self.index_queue = []
class WriteSide:
    def __init__(self, store, cache):
        self.store = store
        self.cache = cache
    def upload(self, tenant, doc_id, text):
        self.store.documents[(tenant, doc_id)] = text
        self.store.versions[tenant] = self.store.versions.get(tenant, 0) + 1
        self.store.index_queue.append((tenant, doc_id))
        self.cache.invalidate_tenant(tenant)
        return self.store.versions[tenant]
class ReadSide:
    def __init__(self, store, cache, llm):
        self.store = store
        self.cache = cache
        self.llm = llm
    def ask(self, tenant, question):
        version = self.store.versions.get(tenant, 0)
        cached = self.cache.get(question, tenant, version)
        if cached is not None:
            return {'answer': cached, 'source': 'cache'}
        docs = [text for (t, _), text in self.store.documents.items() if t == tenant]
        answer = self.llm(question, docs)
        self.cache.put(question, answer, tenant, version)
        return {'answer': answer, 'source': 'model'}
