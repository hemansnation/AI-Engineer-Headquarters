class Registry:
    def __init__(self):
        self.tools = {}
        self.descriptions = {}
    def register(self, name, description):
        def wrap(fn):
            self.tools[name] = fn
            self.descriptions[name] = description
            return fn
        return wrap
    def run(self, name, argument):
        if name not in self.tools:
            return 'Unknown action ' + name + '. Available: ' + ', '.join(self.tools) + ', Finish'
        try:
            return str(self.tools[name](argument))
        except Exception as error:
            return 'Tool error: ' + str(error)
