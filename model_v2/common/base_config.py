class BaseInputerConfig:
    def __init__(self, depot, order, **kwargs):
        self.depot = depot
        self.order = order
        self.kwargs = kwargs


class NegativeSamplingConfig:
    def __init__(self, neg_count=4, neg_col='neg', **kwargs):
        self.neg_count = neg_count
        self.neg_col = neg_col


class BaseConfig:
    def __init__(
            self,
            hidden_size,
            inputer_config,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.inputer_config = BaseInputerConfig(**inputer_config)


class BaseUserConfig(BaseConfig):
    def __init__(
            self,
            negative_sampling=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.negative_sampling = negative_sampling
        if negative_sampling:
            self.negative_sampling = NegativeSamplingConfig(**negative_sampling)

