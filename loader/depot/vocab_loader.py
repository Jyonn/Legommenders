from typing import Dict

from UniTok import Vocab, UniDep

from utils.printer import printer


class VocabLoader:
    def __init__(self):
        self.table = dict()  # type: Dict[str, Vocab]

    def load_from_depot(self, depot: UniDep, order: list):
        for col in order:
            self.table[col] = depot.vocab_depot[depot.get_vocab(col)]

    def register(self, vocab: Vocab, col=None):
        if col:
            self.table[col] = vocab
        else:
            self.table[vocab.name] = vocab

    def __getitem__(self, item):
        return self.table[item]
