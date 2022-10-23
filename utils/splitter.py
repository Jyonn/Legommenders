class Splitter:
    def __init__(self):
        self.order = []
        self.part = dict()

    def add(self, name, weight):
        assert name not in self.order
        assert weight >= 0

        self.order.append(name)
        self.part[name] = weight
        return self

    def divide(self, amount):
        sum_weight = sum(self.part.values())
        assert sum_weight > 0

        range_dict = dict()

        start = 0
        for name in self.order[:-1]:
            end = int(start + self.part[name] / sum_weight * amount) + 1
            range_dict[name] = (start, end)
            start = end

        end = amount
        range_dict[self.order[-1]] = (start, end)
        return range_dict

    def contains(self, name):
        return name in self.part
