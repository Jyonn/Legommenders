import json
import os
import pandas as pd
from collections import defaultdict
import random
from typing import List

from unitok import BertTokenizer, TransformersTokenizer, EntityTokenizer
from unitok.tokenizer.glove_tokenizer import GloVeTokenizer

from processor.base_processor import BaseProcessor, Interactions
from embedder.glove_embedder import GloVeEmbedder
from utils.config_init import ModelInit


random.seed(42)
def neg_sample(all_items, pos_items, n_sample=99) -> List[str]:
    """Sample negative items"""
    pos_set = set(pos_items)  

    max_sample = len(pos_set) + n_sample
    results = [item for item in random.sample(all_items, k=max_sample) if item not in pos_set]
    return results
    


class AmzProcessor(BaseProcessor):
    """Preprocess Videos Games Amazon dataset to 
        create leave-one-out dataset for sequential recommendation
    """
    IID_COL = "iid" # Item ID column name in interaction table
    UID_COL = "uid"
    LBL_COL = "click"
    HIS_COL = "history"

    NEG_COL = "neg"
    NEG_TRUNCATE = 0
    REQUIRE_STRINGIFY = False

    def __init__(self, data_dir=None):
        """
        Args:
            data_dir: Input config for processor.
                This is defined in .data at the Legommenders root folder
                Note: This is not neccessary be a directory. It can be any string
        """
        super().__init__(data_dir)

        interaction_file = os.path.join(self.data_dir, "Video_Games_5.json")

        with open(interaction_file) as fin:
            self.interaction_all = [json.loads(line) for line in fin]

        self.interaction_all.sort(key=lambda r: r["unixReviewTime"])
        metadata_file = os.path.join(self.data_dir, "meta_Video_Games.json")
        with open(metadata_file) as fin:
            self.metadata = [json.loads(line) for line in fin]



    def load_items(self) -> pd.DataFrame:
        """
        Returns: DataFrame used for item. 
            The information must include item ID.
            As for extra information, you can save almost anything here

        For this example, I will save item ID, main_cat (main category) and item title.
        """

        # Load data to dataframe
        df = pd.DataFrame.from_records(
            self.metadata,
            columns=["asin", "main_cat", "title"],
        )

        # Remove item with title
        df = df[df.title != ""]

        # Rename column to our defined names
        return df.rename(columns={
            "asin": self.IID_COL,
            "main_cat": "main_cat",
            "title": "title",
        })

    def load_users(self) -> pd.DataFrame:
        """
        Returns: DataFrame used for user. 
            The information must include:
                - user ID
                - User history
        """

        self.item_set = set(self.item_df[self.IID_COL].unique())

        # Note: Not very good at Pandas so 
        # I will directly write Python code
        users = dict()
        for review in self.interaction_all:
            user_id = review["reviewerID"]
            item_id = review["asin"]
            rating = review["overall"]

            if item_id not in self.item_set:
                continue

            # Implicit feedback dataset --> Ignore rating
            if user_id not in users:
                users[user_id] = {
                    self.UID_COL: user_id,
                    self.HIS_COL: [],
                }

            users[user_id][self.HIS_COL].append(item_id)


        self.user_set = set(users.keys())
        train_inter, val_inter, test_inter, users = self._create_leave_oneout(users)

        self.train_inter_df = pd.DataFrame.from_records(train_inter)
        self.val_inter_df = pd.DataFrame.from_records(val_inter)
        self.test_inter_df = pd.DataFrame.from_records(test_inter)

        return pd.DataFrame.from_records(users)


    def _create_leave_oneout(self, users: dict):
        train_inter = []
        val_inter = []
        test_inter = []

        new_users = []

        # For each user, we need to create
        # three fake user for train, val, test respectively
        for user_id, info in users.items():

            history = info[self.HIS_COL]
            if len(history) < 3:
                print("User", user_id, "has less than 3 interactions")
                continue
            new_users.append({
                self.UID_COL: user_id + "_train",
                self.HIS_COL: history[:-3],
            })
            train_inter.append({
                self.UID_COL: user_id + "_train",
                self.IID_COL: history[-3],
                self.LBL_COL: 1,
            })

            # Val
            new_users.append({
                self.UID_COL: user_id + "_val",
                self.HIS_COL: history[:-2],
            })
            val_inter.append({
                self.UID_COL: user_id + "_val",
                self.IID_COL: history[-2],
                self.LBL_COL: 1,
            })
            neg_vals = neg_sample(self.item_set, history, 198)
            for neg_item in neg_vals[:99]:
                val_inter.append({
                    self.UID_COL: user_id + "_val",
                    self.IID_COL: neg_item,
                    self.LBL_COL: 0,
                })

            # Test
            new_users.append({
                self.UID_COL: user_id + "_test",
                self.HIS_COL: history[:-1],
            })
            test_inter.append({
                self.UID_COL: user_id + "_test",
                self.IID_COL: history[-1],
                self.LBL_COL: 1,
            })
            for neg_item in neg_vals[99:]:
                test_inter.append({
                    self.UID_COL: user_id + "_test",
                    self.IID_COL: neg_item,
                    self.LBL_COL: 0,
                })

        return train_inter, val_inter, test_inter, new_users



    def load_interactions(self) -> Interactions:

        # Note: train, val, and test_inter_df is
        # calculated in load_users already

        return Interactions(
            self.train_inter_df,
            self.val_inter_df,
            self.test_inter_df,
        )

    @property
    def attrs(self) -> dict:
        """
        Returns: Dict[str, int]
            Key is feature name
            Value is maximum length
        """
        return dict(
            title=50,
            main_cat=0, # ID-based feature, no maximum length
        )

    def config_item_tokenization(self):
        """Add list of feature available.
        Here I add Bert, Llama1, Glove
        """

        # Define text feature
        bert_tokenizer = BertTokenizer(vocab='bert')
        llama1_tokenizer = TransformersTokenizer(vocab='llama1', key=ModelInit.get('llama1'))
        glove_tokenizer = GloVeTokenizer(vocab=GloVeEmbedder.get_glove_vocab())

        # The line below create title@bert and main_cat@bert in the feature list.
        self.add_item_tokenizer(bert_tokenizer)

        self.add_item_tokenizer(llama1_tokenizer)
        self.add_item_tokenizer(glove_tokenizer)

        # Add ID-based feature
        self.item.add_feature(tokenizer=EntityTokenizer(vocab='main_cat'), column='main_cat')

    def add_item_tokenizer(self, tokenizer):
        """This function is not required to be defined.
        Here I override the original function as I don't use promp / LLM input
        """
        name = tokenizer.vocab.name

        for attr in self.attrs:
            self.item.add_feature(
                tokenizer=tokenizer,
                column=attr,
                name=f'{attr}@{name}',
                truncate=self.attrs[attr],
            )
