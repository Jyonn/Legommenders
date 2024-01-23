from loader.cacher.base_cacher import BaseCacher
from loader.pager.fast_item_pager import FastItemPager


class ItemCacher(BaseCacher):
    def __init__(self, llm_skip, **kwargs):
        super().__init__(**kwargs)
        self.llm_skip = llm_skip

    def _cache(self, contents):
        item_size = len(contents)
        placeholder = self.operator.get_full_placeholder(item_size)

        pager = FastItemPager(
            inputer=self.operator.inputer,
            contents=contents,
            model=self.operator,
            page_size=self.page_size,
            hidden_size=self.hidden_size,
            llm_skip=self.llm_skip,
            placeholder=placeholder,
        )

        pager.run()

        return pager.fast_item_repr
