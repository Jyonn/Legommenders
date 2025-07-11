"""
repr_cacher.py

High-level façade that orchestrates two independent caches:
    • `ItemCacher` – stores fixed representations for catalogue items
    • `UserCacher` – stores fixed representations for user profiles

Both underlying classes follow the common `BaseCacher` interface
(start / clean / cached flag …).  `ReprCacher` simply wires them
together and forwards the configuration parameters that are needed for
their instantiation.

Why a dedicated wrapper?
------------------------
Downstream code (e.g. the training / evaluation loops) should not care
about the individual cache implementations.  It only needs to
    1) call `repr_cacher.cache(item_contents, user_contents)`
       whenever new contents arrive,
    2) call `repr_cacher.clean()` when the caches must be invalidated
       (e.g. after an embedding layer update).

`ReprCacher` hides all the details and makes the decision whether an
individual cache is active or not (switchable via `activate(False)`).

"""

from __future__ import annotations
from typing import cast

from loader.cacher.item_cacher import ItemCacher
from loader.cacher.user_cacher import UserCacher
from loader.env import Env


class ReprCacher:
    """
    Wrapper that bundles an `ItemCacher` and a `UserCacher`.

    Parameters
    ----------
    legommender : Legommender
        Main model object that owns the operators / configs needed for
        the individual cachers.  The import is *local* to avoid circular
        dependencies (“chicken-&-egg” problem during module loading).
    """

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self, legommender):
        # Post-pone import to runtime to prevent circular import issues
        from model.legommender import Legommender

        # Provide the type information for static checkers / IDEs
        legommender = cast(Legommender, legommender)

        # Convenience handle
        config = legommender.config

        # Whether item contents (text / meta data) are used at all
        self.use_item_content: bool = config.use_item_content

        # Total number of users (tokeniser vocab size = user ids)
        self.user_size: int = (
            config.user_ut.meta.features[legommender.cm.user_col]
            .tokenizer.vocab.size
        )

        # Global on/off switch – can be toggled via `activate()`
        self._activate: bool = True

        # --------------------------------------------------------------
        # Instantiate the two sub-cachers
        # --------------------------------------------------------------
        # 1) Item cache
        self.item: ItemCacher = ItemCacher(
            operator=legommender.item_op,
            page_size=config.cache_page_size,
            hidden_size=config.hidden_size,
            # Only active when (a) an item operator exists and (b) it
            # explicitly allows caching
            activate=(
                legommender.item_op is not None
                and legommender.item_op.allow_caching
            ),
            trigger=Env.set_item_cache,  # pass the resulting tensor to Env
        )

        # 2) User cache
        self.user: UserCacher = UserCacher(
            operator=legommender.get_user_content,  # callable that builds all user features
            page_size=config.cache_page_size,
            hidden_size=config.hidden_size,
            activate=legommender.user_op.allow_caching,
            # Pre-allocated destination passed directly (frees the pager
            # from allocating it on its own)
            placeholder=legommender.user_op.get_full_placeholder(
                self.user_size
            ),
            trigger=Env.set_user_cache,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def activate(self, activate: bool) -> None:
        """
        Enable / disable both sub-cachers at once.

        When de-activated `cache()` becomes a no-op, but the internal
        state of the sub-cachers is untouched.
        """
        self._activate = activate

    def cache(self, item_contents, user_contents) -> None:
        """
        Build / refresh the caches for the supplied contents.

        Parameters
        ----------
        item_contents : list[Any]
            List of items fed into `ItemCacher`.  May be ignored when
            `self.use_item_content == False`.
        user_contents : list[Any]
            List of user features fed into `UserCacher`.
        """
        if not self._activate:
            return

        # Only cache items when item contents are part of the model
        if self.use_item_content:
            self.item.cache(item_contents)

        # User cache is always updated
        self.user.cache(user_contents)

    def clean(self) -> None:
        """
        Invalidate / remove both caches.
        """
        self.item.clean()
        self.user.clean()
