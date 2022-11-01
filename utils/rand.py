import random
import string


class Rand(dict):
    chars = string.ascii_letters + string.digits

    def __init__(self):
        dict.__init__(self, {})

    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        return ''.join([random.choice(self.chars) for _ in range(int(item))])

    def __str__(self):
        return '<Random class>'

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    print(Rand()[4])
