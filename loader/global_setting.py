class Setting:
    # running mode
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'
    MODES = [TRAIN, DEV, TEST]

    # device
    device = None
    status = None

    # pad
    UNSET = -1
