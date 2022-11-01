import datetime


class Timing(dict):
    def __init__(self):
        dict.__init__(self, {})

    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        now = datetime.datetime.now()
        if item == 'str':
            return now.strftime('%y%m%d-%H%M%S')
        else:
            return hex(int(now.timestamp()))[2:]

    def __str__(self):
        return '<Random class>'

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    print(Timing()['str'])
