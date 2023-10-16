from loader.global_setting import Setting
from utils.pagers.fast_doc_pager import FastDocPager
from utils.pagers.fast_user_pager import FastUserPager
from utils.printer import printer, Color


class Cacher:
    def __init__(self, recommender):
        self.recommender = recommender
        self.use_news_content = self.recommender.config.use_item_content
        self.news_encoder = self.recommender.item_encoder
        self.user_encoder = self.recommender.user_encoder
        self.llm_skip = self.recommender.llm_skip

        self.fast_doc_eval = False
        self.fast_doc_repr = None
        self.use_fast_doc_caching = True

        self.fast_user_eval = False
        self.fast_user_repr = None
        self.use_fast_user_caching = True

        self.print = printer[(self.__class__.__name__, '|', Color.MAGENTA)]

    def start_caching_doc_repr(self, doc_list):
        if not self.use_news_content:
            return
        if not Setting.fast_eval:
            return
        if not self.use_fast_doc_caching:
            return
        if self.fast_doc_eval:
            return

        self.print("Start caching doc repr")

        pager = FastDocPager(
            inputer=self.news_encoder.inputer,
            contents=doc_list,
            model=self.news_encoder,
            page_size=self.recommender.config.page_size,
            hidden_size=self.recommender.config.hidden_size,
            llm_skip=self.llm_skip,
        )

        pager.run()

        self.fast_doc_eval = True
        self.fast_doc_repr = pager.fast_doc_repr

    def end_caching_doc_repr(self):
        self.fast_doc_eval = False
        self.fast_doc_repr = None

    def start_caching_user_repr(self, user_list):
        if not Setting.fast_eval:
            return
        if not self.use_fast_user_caching:
            return
        if self.fast_user_eval:
            return

        self.print("Start caching user repr")

        pager = FastUserPager(
            contents=user_list,
            model=self.recommender.get_user_content,
            page_size=self.recommender.config.page_size,
            hidden_size=self.recommender.config.hidden_size,
        )

        pager.run()

        self.fast_user_eval = True
        self.fast_user_repr = pager.fast_user_repr

    def end_caching_user_repr(self):
        self.fast_user_eval = False
        self.fast_user_repr = None
