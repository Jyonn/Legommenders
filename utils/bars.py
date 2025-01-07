from tqdm import tqdm


class Unset:
    pass


class Bar:

    def __init__(self):
        self.iterable = Unset
        self.bar_format = Unset
        self.leave = Unset
        self.kwargs = dict()

    def is_filled(self):
        for key in self.__dict__:
            if self.__dict__[key] is Unset:
                return False
        return True

    def get_config(self):
        kwargs = dict()
        kwargs.update(self.kwargs)
        for key in self.__dict__:
            if key != "kwargs":
                kwargs[key] = self.__dict__[key]
        return kwargs

    def __call__(self, iterable=None, **kwargs):
        if iterable is not None:
            kwargs["iterable"] = iterable

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        if self.is_filled():
            return tqdm(**self.get_config())


class TrainBar(Bar):
    def __init__(self, epoch):
        super().__init__()

        self.bar_format = "Training Epoch {epoch} [{percentage:.0f}% < {remaining}] {postfix}"
        self.bar_format = self.bar_format.replace("{epoch}", str(epoch))
        self.leave = False


class DevBar(Bar):
    def __init__(self, epoch, train_loss):
        super().__init__()

        self.bar_format = "Train loss: {train_loss} | Validating Epoch {epoch} [{percentage:.0f}% < {remaining}] {postfix}"
        self.bar_format = self.bar_format.replace("{epoch}", str(epoch))
        self.bar_format = self.bar_format.replace("{train_loss}", f"{train_loss:.4f}")
        self.leave = False


class DescBar(Bar):
    def __init__(self, desc):
        super().__init__()

        self.bar_format = "{desc} [{percentage:.0f}% < {remaining}]"
        self.bar_format = self.bar_format.replace("{desc}", desc)
        self.leave = False

#
# def train(iterable, epoch, **kwargs):
#     bar = "Training Epoch {epoch} [{percentage:.0f}% < {remaining}] {postfix}"
#     bar = bar.replace("{epoch}", str(epoch))
#     return tqdm(
#         iterable=iterable,
#         bar_format=bar,
#         ncols=100,
#         leave=False,
#         **kwargs,
#     )
#
#
# def dev(iterable, epoch, train_loss, **kwargs):
#     bar = "Train loss: {train_loss} | Validating Epoch {epoch} [{percentage:.0f}% < {remaining}] {postfix}"
#     bar = bar.replace("{epoch}", str(epoch))
#     bar = bar.replace("{train_loss}", f"{train_loss:.4f}")
#     return tqdm(
#         iterable=iterable,
#         bar_format=bar,
#         ncols=100,
#         leave=False,
#         **kwargs,
#     )
