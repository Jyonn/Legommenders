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
    llm_cache = False

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
    def set_llm_cache(cls, llm_cache):
        cls.llm_cache = llm_cache
