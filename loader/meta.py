class DatasetType:
    news = 'news'
    book = 'book'


class Phases:
    train = 'train'
    dev = 'dev'
    test = 'test'
    fast_eval = 'fast_eval'


class Meta:
    # device
    device = None
    simple_dev = False

    # pad
    UNSET = -1

    # dataset
    data_type = None
