from model.inputer.flatten_seq_inputer import FlattenSeqInputer
from model.operators.fastformer_operator import FastformerOperator


class FlattenFastformerOperator(FastformerOperator):
    inputer_class = FlattenSeqInputer
    flatten_mode = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.target_user, 'flatten transformer operator is only designed as user encoder'
