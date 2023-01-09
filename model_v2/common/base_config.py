class BaseInputerConfig:
    def __init__(self, depot, order, **kwargs):
        self.depot = depot
        self.order = order
        self.kwargs = kwargs


class BaseConfig:
    def __init__(
            self,
            hidden_size,
            inputer_config,
            **kwargs,
    ):
        self.hidden_size = hidden_size

