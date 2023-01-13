import random
import string
import time
from hashlib import md5


class Rand(dict):
    chars = string.ascii_letters + string.digits

    def __init__(self):
        dict.__init__(self, {})

    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        crt_time = str(time.time())
        crt_time = md5(crt_time.encode('utf-8')).hexdigest()
        return ''.join([random.choice(crt_time) for _ in range(int(item))])

    def __str__(self):
        return '<Random class>'

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':

    d = dict(
        utils=dict(
            rand=Rand()
        ),
        store=dict(
            path='${utils.rand.4}',
            filename='${store.path}/exp.log',
        )
    )

    import smartdict
    print(smartdict.parse(d))

