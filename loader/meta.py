from unitok import Symbol


class LegoSymbols:
    news = Symbol('news')
    book = Symbol('book')

    train = Symbol('train')
    dev = Symbol('dev')
    test = Symbol('test')
    fast_eval = Symbol('fast_eval')


class Meta:
    # device
    device = None
    simple_dev = False

    # pad
    UNSET = -1

    # dataset
    data_type = None
