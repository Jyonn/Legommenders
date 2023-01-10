class Meaner:
    def __init__(self):
        self._list = []

    def add(self, value):
        self._list.append(value)

    def mean(self):
        return sum(self._list) / len(self._list)
