class Meaner:
    def __init__(self):
        self.mean = 0
        self.count = 0

    def add(self, value):
        self.mean = (self.mean * self.count + value) / (self.count + 1)
        self.count += 1

    def __call__(self, value):
        self.add(value)
        return self.mean
