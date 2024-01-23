from loader.cacher.base_cacher import BaseCacher
from loader.pager.fast_user_pager import FastUserPager


class UserCacher(BaseCacher):
    def __init__(self, placeholder, **kwargs):
        super().__init__(**kwargs)
        self.placeholder = placeholder

    def _cache(self, contents):
        pager = FastUserPager(
            contents=contents,
            model=self.operator,
            page_size=self.page_size,
            # page_size=2,
            hidden_size=self.hidden_size,
            placeholder=self.placeholder,
        )
        pager.run()
        return pager.fast_user_repr
