from loader.cacher.item_cacher import ItemCacher
from loader.cacher.user_cacher import UserCacher


class ReprCacher:
    def __init__(self, legommender):
        self.use_item_content = legommender.config.use_item_content
        self._activate = True

        self.item = ItemCacher(
            operator=legommender.item_encoder,
            page_size=legommender.config.page_size,
            hidden_size=legommender.config.hidden_size,
            llm_skip=legommender.llm_skip,
        )

        self.user = UserCacher(
            operator=legommender.get_user_content,
            page_size=legommender.config.page_size,
            hidden_size=legommender.config.hidden_size,
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
