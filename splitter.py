from oba import Obj

from base_lego import BaseLego
from loader.pager.llm_split_pager import LLMSplitPager
from model.operators.base_llm_operator import BaseLLMOperator
from utils.config_init import CommandInit


class Splitter(BaseLego):
    def run(self):
        item_op = self.legommender.item_op
        if not isinstance(item_op, BaseLLMOperator):
            raise ValueError('item encoder is not a LLM operator')
        if not self.embed.embeddings:
            raise ValueError('please specify pretrained embedding configurations when using LLM layer split')

        pager = LLMSplitPager(
            inputer=item_op.inputer,
            layers=Obj.raw(self.exp.store.layers),
            hidden_size=item_op.config.input_dim,
            contents=self.resampler.item_cache,
            model=item_op.get_all_hidden_states,
            page_size=self.manager.lego_config.item_page_size,
        )

        pager.run()
        pager.store(self.exp.store.dir)


if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model', 'exp', 'embed'],
        default_args=dict(
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
        ),
    ).parse()

    splitter = Splitter(config=configuration)
    splitter.run()
