import abc

from transformers import BertModel

from model.operators.iisan_operator import IISANOperator


class BertIISANOperator(IISANOperator, abc.ABC):
    transformer: BertModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer.embeddings.word_embeddings = None
        self._load_hidden_states()

        if self.transformer.config.hidden_size != self.config.input_dim:
            raise ValueError(f'In {self.classname}, hidden_size of transformer ({self.transformer.config.hidden_size}) '
                             f'does not match input_dim ({self.config.input_dim})')


class BertBaseIISANOperator(BertIISANOperator):
    pass


class BertLargeIISANOperator(BertIISANOperator):
    pass
