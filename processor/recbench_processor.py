import os.path

import pandas as pd
import yaml
from unitok import BertTokenizer, TransformersTokenizer, GloVeTokenizer

from embedder.glove_embedder import GloVeEmbedder
from processor.base_processor import BaseProcessor, Interactions
from utils.config_init import ModelInit


class RecBenchProcessor(BaseProcessor):
    PROMPT: str
    NEG_COL: str = 'neg'
    NEG_TRUNCATE = 100

    BASE_STORE_DIR = os.path.join('data', 'recbench')

    def __init__(self, data_dir):
        super().__init__(data_dir=data_dir)

        self.item_df = pd.read_parquet(os.path.join(data_dir, 'items.parquet'))
        self.user_df = pd.read_parquet(os.path.join(data_dir, 'users.parquet'))
        self.finetune_df = pd.read_parquet(os.path.join(data_dir, 'finetune.parquet'))
        self.test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))

        self.user_df[self.HIS_COL] = self.user_df[self.HIS_COL].apply(
            lambda x: x if isinstance(x, list) else x.tolist())

        self.valid_user_set = self.load_valid_user_set(valid_ratio=0.1)
        self.train_df = self.finetune_df[~self.finetune_df[self.UID_COL].isin(self.valid_user_set)]
        self.valid_df = self.finetune_df[self.finetune_df[self.UID_COL].isin(self.valid_user_set)]

    def config_item_tokenization(self):
        bert_tokenizer = BertTokenizer(vocab='bert')
        llama1_tokenizer = TransformersTokenizer(vocab='llama1', key=ModelInit.get('llama1'))
        glove_tokenizer = GloVeTokenizer(vocab=GloVeEmbedder.get_glove_vocab())

        self.add_item_tokenizer(bert_tokenizer)
        self.add_item_tokenizer(llama1_tokenizer)
        self.add_item_tokenizer(glove_tokenizer)

    def load_valid_user_set(self, valid_ratio: float) -> set:
        with open(os.path.join(self.data_dir, f'valid_user_set_{valid_ratio}.txt'), 'r') as f:
            return {line.strip() for line in f}

    def load_items(self) -> pd.DataFrame:
        for attr in self.attrs:
            # if is empty, set to "empty"
            self.item_df[attr] = self.item_df[attr].fillna('[empty]')

        self.item_df['prompt'] = self.PROMPT
        for attr in self.attrs:
            self.item_df[f'prompt_{attr}'] = attr.upper()[0] + attr[1:].lower() + ': '
        return self.item_df

    def load_users(self) -> pd.DataFrame:
        return self.user_df

    def load_interactions(self) -> Interactions:
        return Interactions(self.train_df, self.valid_df, self.test_df)

    def generate_data_configuration(self):
        return dict(
            name=self.get_name(),
            item=dict(
                ut=self.item_save_dir,
                inputs=[attr + '@${lm}' for attr in self.attrs],
            ),
            user=dict(
                ut=self.user_save_dir,
                truncate=50
            ),
            inter=dict(
                train=self.get_save_dir(Interactions.train),
                dev=self.get_save_dir(Interactions.valid),
                test=self.get_save_dir(Interactions.test),
                filters=dict(
                    history=['lambda x: x']
                )
            ),
            column_map=dict(
                item_col=self.IID_FEAT,
                user_col=self.UID_FEAT,
                history_col=self.HIS_FEAT,
                neg_col=self.NEG_COL,
                label_col=self.LBL_FEAT,
                group_col=self.UID_FEAT,
            )
        )

    def load(self, regenerate=False):
        super().load(regenerate=regenerate)

        yaml_path = os.path.join('config', 'data', f'{self.get_name()}.yaml')

        if not os.path.exists(yaml_path) or regenerate:
            data_config = self.generate_data_configuration()
            with open(yaml_path, 'w') as f:
                yaml.dump(data_config, f)

            print(f'Data configuration saved to {yaml_path}, please use `python trainer.py --data {yaml_path} --lm ...` to train')
