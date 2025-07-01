import os
from typing import cast

import pandas as pd
from pigmento import pnt
from unitok import UniTok, EntityTokenizer, TransformersTokenizer, Job

from processor.base_processor import BaseProcessor
from processor.mind_processor import MINDProcessor
from utils.config_init import ModelInit


class XMINDProcessor(BaseProcessor):
    IID_COL = 'nid'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizers = [
            TransformersTokenizer(vocab='llama1', key=ModelInit.get('llama1'))
        ]

    @property
    def attrs(self) -> dict:
        return dict(
            title=50,
            abstract=200,
        )

    @staticmethod
    def load_base_item():
        mind_dir = os.path.join(MINDProcessor.BASE_STORE_DIR, MINDProcessor.get_name())
        if not os.path.exists(os.path.join(mind_dir, 'items')):
            raise ValueError(
                f'xMINDProcessor requires the processed MIND dataset. Please run `python process.py --data mind` first.')
        return UniTok.load(os.path.join(mind_dir, 'items'))

    def config_item_tokenization(self):
        for tokenizer in self.tokenizers:
            name = tokenizer.vocab.name
            for attr in self.attrs:
                self.item.add_feature(tokenizer=tokenizer, column=attr, name=f'{attr}@{name}', truncate=self.attrs[attr])

    def _load_items(self, path: str) -> pd.DataFrame:
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            names=[self.IID_COL, 'title', 'abstract'],
        )

    def load_items(self):
        train_df = self._load_items(os.path.join(self.data_dir, 'xMINDsmall_train', 'news.tsv'))
        valid_df = self._load_items(os.path.join(self.data_dir, 'xMINDsmall_dev', 'news.tsv'))
        item_df = pd.concat([train_df, valid_df]).drop_duplicates([self.IID_COL])
        item_df['abstract'] = item_df['abstract'].fillna('')
        return item_df

    def load(self, regenerate=False):
        pnt(f'load {self.get_name()} processor')

        if not regenerate and os.path.exists(self.item_save_dir):
            self.item = UniTok.load(self.item_save_dir)
            return

        base_item: UniTok = self.load_base_item()
        remove_features = []
        for feature in base_item.meta.features:  # type: Job
            if feature is base_item.key_feature:
                continue
            if '@' in feature.name:
                remove_features.append(feature)
        for feature in remove_features:
            base_item.remove_feature(feature)

        self.item_df = self.load_items()

        item_vocab = base_item.key_feature.tokenizer.vocab
        item_vocab.deny_edit()

        self.item_df = self.item_df[self.item_df[self.IID_COL].isin(item_vocab.o2i)]
        self.item_df = self.item_df.sort_values(by=self.IID_COL, key=lambda o: o.map(item_vocab.o2i))
        self.item_df = self.item_df.reset_index(drop=True)

        assert len(self.item_df) == len(item_vocab), 'xMIND items should be fully aligned with the base MIND items.'

        self.item_df.to_parquet(os.path.join(self.save_dir, f'items.parquet'))

        with UniTok() as self.item:
            self.item.add_feature(tokenizer=EntityTokenizer(vocab=item_vocab), column=self.IID_COL, name=self.IID_FEAT, key=True)
            self.config_item_tokenization()
            self.item.tokenize(self.item_df)
            self.item.union(base_item, soft_union=False)
            self.item.save(self.item_save_dir)
            pnt(f'tokenized {len(self.item)} items')


class XMINDCMNProcessor(XMINDProcessor): ...
class XMINDFINProcessor(XMINDProcessor): ...
class XMINDGRNProcessor(XMINDProcessor): ...
class XMINDHATProcessor(XMINDProcessor): ...
class XMINDINDProcessor(XMINDProcessor): ...
class XMINDJPNProcessor(XMINDProcessor): ...
class XMINDRONProcessor(XMINDProcessor): ...
class XMINDSOMProcessor(XMINDProcessor): ...
class XMINDSWHProcessor(XMINDProcessor): ...
class XMINDTAMProcessor(XMINDProcessor): ...
class XMINDTHAProcessor(XMINDProcessor): ...
class XMINDTURProcessor(XMINDProcessor): ...
class XMINDVIEProcessor(XMINDProcessor): ...
