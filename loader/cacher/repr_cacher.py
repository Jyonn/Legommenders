from typing import cast

from loader.cacher.item_cacher import ItemCacher
from loader.cacher.user_cacher import UserCacher
from loader.env import Env


class ReprCacher:
    def __init__(self, legommender):
        from model.legommender import Legommender
        legommender = cast(Legommender, legommender)

        self.use_item_content = legommender.config.use_item_content
        self.user_size = legommender.user_hub.ut.meta.jobs[legommender.cm.user_col].tokenizer.vocab.size
        self._activate = True

        self.item = ItemCacher(
            operator=legommender.item_op,
            page_size=legommender.config.cache_page_size,
            hidden_size=legommender.config.hidden_size,
            llm_skip=legommender.llm_skip,
            activate=legommender.item_op and legommender.item_op.allow_caching,
            trigger=Env.set_item_cache,
        )

        self.user = UserCacher(
            operator=legommender.get_user_content,
            page_size=legommender.config.cache_page_size,
            hidden_size=legommender.config.hidden_size,
            activate=legommender.user_op.allow_caching,
            placeholder=legommender.user_op.get_full_placeholder(self.user_size),
            trigger=Env.set_user_cache,
        )

    def activate(self, activate):
        self._activate = activate

    def cache(self, item_contents, user_contents):
        if not self._activate:
            return

        if self.use_item_content:
            self.item.cache(item_contents)

        self.user.cache(user_contents)

    def clean(self):
        self.item.clean()
        self.user.clean()
