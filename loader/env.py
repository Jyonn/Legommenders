from utils.path_hub import PathHub
from utils.timer import Timer


class Env:
    # device
    device = None
    simple_dev = False

    # pad
    UNSET = -1

    # status
    is_training = True
    is_evaluating = False
    is_testing = False

    # resampler
    item_cache = False
    user_cache = False
    lm_cache = False

    # timer
    timer = Timer()  # used for debug
    latency_timer = Timer()  # used for latency

    @classmethod
    def train(cls):
        cls.is_training = True
        cls.is_evaluating = False
        cls.is_testing = False

    @classmethod
    def dev(cls):
        cls.is_training = False
        cls.is_evaluating = True
        cls.is_testing = False

    @classmethod
    def test(cls):
        cls.is_training = False
        cls.is_evaluating = False
        cls.is_testing = True

    @classmethod
    def set_device(cls, device):
        cls._device = device

    @classmethod
    def set_item_cache(cls, item_cache):
        cls.item_cache = item_cache

    @classmethod
    def set_user_cache(cls, user_cache):
        cls.user_cache = user_cache

    @classmethod
    def set_lm_cache(cls, lm_cache):
        cls.lm_cache = lm_cache

    @classmethod
    def set_path_hub(cls, path_hub: PathHub):
        cls.path_hub = path_hub
