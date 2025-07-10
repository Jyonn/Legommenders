"""
progress_bar.py

Wrapper utilities around *tqdm* that make it more convenient to create
consistently-looking progress bars for various training phases
(train/validation/test) without repeating boiler-plate code.

Key concepts
------------
Unset
    A simple sentinel object used to indicate that an attribute has not
    yet been configured.

Bar
    A tiny “configuration holder” that defers the construction of a tqdm
    object until *all* mandatory attributes have been assigned.  The
    first time the instance receives enough information, it returns an
    actual `tqdm` iterator.

Specialised subclasses
----------------------
TrainBar(epoch)
    Progress bar for the training loop of a given *epoch*.

DevBar(epoch, train_loss)
    Progress bar for the validation loop; also displays the last training
    loss value for quick reference.

TestBar()
    Generic progress bar for the test / inference phase.

DescBar(desc)
    A free-form progress bar with a custom textual prefix.
"""

from typing import Any, Dict, Optional, Iterable

from tqdm import tqdm


# =============================================================================
#                                   Sentinel
# =============================================================================
class Unset:
    """
    A unique marker used to distinguish “not yet configured” attributes
    from attributes that have legitimately been set to `None` or `False`.
    """
    pass


# =============================================================================
#                                    Bar
# =============================================================================
class Bar:
    """
    A *lazy* progress-bar builder.

    Typical usage
    -------------
    >>> bar = TrainBar(epoch=3)                 # partially configured
    >>> for batch in bar(dataloader):           # first call constructs tqdm
    ...     ...

    Attributes set directly on the object (either via constructor in a
    subclass or via keyword arguments passed to `__call__`) are kept in
    `self.__dict__`.  Once every required field is filled
    (`is_filled() == True`) the actual tqdm object is created.
    """

    def __init__(self):
        # Core tqdm arguments (initially unset)
        self.iterable: Any = Unset
        self.bar_format: str = Unset
        self.leave: bool = Unset

        # Arbitrary extra kwargs forwarded to tqdm
        self.kwargs: Dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def is_filled(self) -> bool:
        """Return True if *all* predefined attributes have been set."""
        for key in self.__dict__:
            if self.__dict__[key] is Unset:
                return False
        return True

    def get_config(self) -> Dict[str, Any]:
        """
        Assemble a dictionary of parameters to be passed to `tqdm(...)`.
        Both the explicitly declared attributes and the user-supplied
        `kwargs` are included.
        """
        cfg = dict(self.kwargs)  # start with arbitrary kwargs
        for key, value in self.__dict__.items():
            if key != "kwargs":
                cfg[key] = value
        return cfg

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def __call__(self, iterable: Optional[Iterable] = None, **kwargs):
        """
        Either *update* the internal configuration or, if everything is
        ready, instantiate and return a real tqdm object.

        Parameters
        ----------
        iterable : Iterable | None
            The sequence to iterate over.  May also be provided earlier
            via subclass constructor.
        **kwargs
            Any additional tqdm parameters (total, ncols, …).  Unknown
            keys are forwarded untouched.
        """
        # Allow passing the iterable positionally
        if iterable is not None:
            kwargs["iterable"] = iterable

        # Update configuration
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Unknown keys are treated as plain tqdm kwargs
                self.kwargs[key] = value

        # If fully configured → build tqdm
        if self.is_filled():
            return tqdm(**self.get_config())


# =============================================================================
#                     Convenience sub-classes for common phases
# =============================================================================
class TrainBar(Bar):
    """Pre-configured progress bar for the *training* loop."""

    def __init__(self, epoch: int):
        super().__init__()
        self.bar_format = (
            "Training Epoch {epoch} [{percentage:.0f}% < {remaining}] {postfix}"
        ).replace("{epoch}", str(epoch))
        self.leave = False  # do not keep bar after completion


class DevBar(Bar):
    """Progress bar for the *validation* / dev loop."""

    def __init__(self, epoch: int, train_loss: float):
        super().__init__()
        self.bar_format = (
            "Train loss: {train_loss} | "
            "Validating Epoch {epoch} [{percentage:.0f}% < {remaining}] {postfix}"
        )
        self.bar_format = self.bar_format.replace("{epoch}", str(epoch)).replace(
            "{train_loss}", f"{train_loss:.4f}"
        )
        self.leave = False


class TestBar(Bar):
    """Progress bar for the *test* / inference phase."""

    def __init__(self):
        super().__init__()
        self.bar_format = "Testing [{percentage:.0f}% < {remaining}] {postfix}"
        self.leave = False


class DescBar(Bar):
    """
    Generic progress bar with a user-defined description in front
    (useful for ad-hoc loops).
    """

    def __init__(self, desc: str):
        super().__init__()
        self.bar_format = "{desc} [{percentage:.0f}% < {remaining}]".replace(
            "{desc}", desc
        )
        self.leave = False
