from typing import cast

from loader.cacher.item_cacher import ItemCacher
from loader.cacher.user_cacher import UserCacher


class ReprCacher:
    def __init__(self, legommender):
        from model.legommender import Legommender
        legommender = cast(Legommender, legommender)

        self.use_item_content = legommender.config.use_item_content
        self.user_size = legommender.user_hub.depot.vocs[legommender.column_map.user_col].size
        self._activate = True

        self.item = ItemCacher(
            operator=legommender.item_encoder,
            page_size=legommender.config.page_size,
            hidden_size=legommender.config.hidden_size,
            llm_skip=legommender.llm_skip,
            activate=legommender.item_encoder and legommender.item_encoder.allow_caching,
        )

        self.user = UserCacher(
            operator=legommender.get_user_content,
            page_size=legommender.config.page_size,
            hidden_size=legommender.config.hidden_size,
            activate=legommender.user_encoder.allow_caching,
            placeholder=legommender.user_encoder.get_full_placeholder(self.user_size),
        )

        self.user_plugin = legommender.user_plugin

    def activate(self, activate):
        self._activate = activate

    def cache(self, item_contents, user_contents):
        if not self._activate:
            return

        if self.use_item_content:
            self.item.cache(item_contents)

        if self.user_plugin:
            self.user_plugin.cache()
        self.user.cache(user_contents)

    def clean(self):
        if self.user_plugin:
            self.user_plugin.clean()

        self.item.clean()
        self.user.clean()
