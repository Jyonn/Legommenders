from loader.cacher.base_cacher import BaseCacher
from loader.pager.fast_user_pager import FastUserPager


class UserCacher(BaseCacher):
    def _cache(self, contents):
        pager = FastUserPager(
            contents=contents,
            model=self.operator,
            page_size=self.page_size,
            # page_size=2,
            hidden_size=self.hidden_size,
        )
        pager.run()
        return pager.fast_user_repr
