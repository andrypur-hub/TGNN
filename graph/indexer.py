class NodeIndexer:
    def __init__(self):
        self.map = {}
        self.counter = 0

    def get(self, key):
        if key not in self.map:
            self.map[key] = self.counter
            self.counter += 1
        return self.map[key]
