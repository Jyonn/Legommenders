"""
xmind_processor.py

Processors for multilingual xMIND datasets
------------------------------------------

xMIND is an extension of the original MIND dataset where the English
articles have been *machine-translated* into various target languages.
Each language gets its own tiny subset such as `xMINDsmall_train/`.

The processing pipeline below **reuses** the already tokenized English
MIND catalogue so that
    • the item vocabulary (news-ids) stays **identical**,
    • we only need to re-tokenize the *textual* attributes for the target
      language,
    • all ID-to-index mappings remain aligned across languages, making
      cross-lingual experiments straightforward.

Only the code comments are added / extended – the executable logic
remains unchanged.
"""

import os
from typing import cast

import pandas as pd
from pigmento import pnt
from unitok import UniTok, EntityTokenizer, TransformersTokenizer, Job

from processor.base_processor import BaseProcessor
from processor.mind_processor import MINDProcessor
from utils.config_init import ModelInit


# ────────────────────────────────────────────────────────────────────────
#                               Base class
# ────────────────────────────────────────────────────────────────────────
class XMINDProcessor(BaseProcessor):
    """
    Generic processor that handles one *language* of xMIND.
    Concrete language subclasses listed at the bottom merely inherit the
    behaviour without any modifications.
    """

    IID_COL = "nid"  # news identifier shared with the English MIND set

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We only tokenize with LLaMA-1 in this example; easily extendable.
        self.tokenizers = [
            TransformersTokenizer(vocab="llama1", key=ModelInit.get("llama1"))
        ]

    # ---------------- textual attribute spec ------------------------
    @property
    def attrs(self) -> dict:
        # attr_name -> max_token_length
        return dict(
            title=50,
            abstract=200,
        )

    # ---------------- helper: load base English catalogue -----------
    @staticmethod
    def load_base_item() -> UniTok:
        """
        Load the already processed **English** MIND items UniTok.  Required
        for vocab alignment.  Raises if not found.
        """
        mind_dir = os.path.join(MINDProcessor.BASE_STORE_DIR, MINDProcessor.get_name())
        if not os.path.exists(os.path.join(mind_dir, "items")):
            raise ValueError(
                "xMINDProcessor requires the processed MIND dataset. "
                "Please run `python process.py --data mind` first."
            )
        return UniTok.load(os.path.join(mind_dir, "items"))

    # ---------------- item tokenization config ----------------------
    def config_item_tokenization(self):
        # Register every LLM tokenizer for each attribute
        for tokenizer in self.tokenizers:
            name = tokenizer.vocab.name
            for attr in self.attrs:
                self.item.add_feature(
                    tokenizer=tokenizer,
                    column=attr,
                    name=f"{attr}@{name}",
                    truncate=self.attrs[attr],
                )

    # ---------------- raw TSV loader --------------------------------
    def _load_items(self, path: str) -> pd.DataFrame:
        """
        TSV has only three columns: nid, title, abstract.
        """
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep="\t",
            names=[self.IID_COL, "title", "abstract"],
        )

    def load_items(self) -> pd.DataFrame:
        """
        Merge train + dev news files, drop duplicates and ensure abstract
        is non-null so that tokenizers always receive a string.
        """
        train_df = self._load_items(os.path.join(self.data_dir, "xMINDsmall_train", "news.tsv"))
        valid_df = self._load_items(os.path.join(self.data_dir, "xMINDsmall_dev", "news.tsv"))
        item_df = pd.concat([train_df, valid_df]).drop_duplicates([self.IID_COL])
        item_df["abstract"] = item_df["abstract"].fillna("")
        return item_df

    # =================================================================
    #                       Custom load routine
    # =================================================================
    def load(self, regenerate: bool = False):
        """
        Only items need processing (no users / interactions in xMINDsmall).
        If an items UniTok already exists and `regenerate` is False we skip
        heavy lifting and just load the object.
        """
        pnt(f"load {self.get_name()} processor")

        # Fast path: load cached UniTok ---------------------------------
        if not regenerate and os.path.exists(self.item_save_dir):
            self.item = UniTok.load(self.item_save_dir)
            return

        # Slow path: rebuild -------------------------------------------
        base_item: UniTok = self.load_base_item()

        # Remove *all* non-key features that came from previous tokenizers;
        # we will add language-specific ones again.
        remove_features = []
        for feature in base_item.meta.features:  # type: Job
            if feature is base_item.key_feature:
                continue
            if "@" in feature.name:  # textual features use "@tokenizer" suffix
                remove_features.append(feature)
        for feature in remove_features:
            base_item.remove_feature(feature)

        # Load translated news articles
        self.item_df = self.load_items()

        # Ensure perfect alignment between English & translated nids ----
        item_vocab = base_item.key_feature.tokenizer.vocab
        item_vocab.deny_edit()  # freeze

        self.item_df = self.item_df[self.item_df[self.IID_COL].isin(item_vocab.o2i)]
        self.item_df = self.item_df.sort_values(
            by=self.IID_COL, key=lambda s: s.map(item_vocab.o2i)
        ).reset_index(drop=True)

        assert len(self.item_df) == len(
            item_vocab
        ), "xMIND items should be fully aligned with the base MIND items."

        # Keep a copy of the raw parquet for debugging / reuse ----------
        self.item_df.to_parquet(os.path.join(self.save_dir, "items.parquet"))

        # Build new UniTok object --------------------------------------
        with UniTok() as self.item:
            # Key feature is still the shared *nid* vocab
            self.item.add_feature(
                tokenizer=EntityTokenizer(vocab=item_vocab),
                column=self.IID_COL,
                name=self.IID_FEAT,
                key=True,
            )
            self.config_item_tokenization()

            # Tokenise translated text
            self.item.tokenize(self.item_df)

            # Merge with the English baseline (keeps identical ids)
            self.item.union(base_item, soft_union=False)

            self.item.save(self.item_save_dir)
            pnt(f"tokenized {len(self.item)} items")


# ────────────────────────────────────────────────────────────────────────
#                        Language-specific subclasses
# ────────────────────────────────────────────────────────────────────────
# Each of these simply inherits XMINDProcessor so that users can call
# `python process.py --data xmindjpn` etc.
class XMINDCMNProcessor(XMINDProcessor): ...  # Chinese
class XMINDFINProcessor(XMINDProcessor): ...  # Finnish
class XMINDGRNProcessor(XMINDProcessor): ...  # Guarani
class XMINDHATProcessor(XMINDProcessor): ...  # Haitian
class XMINDINDProcessor(XMINDProcessor): ...  # Indonesian
class XMINDJPNProcessor(XMINDProcessor): ...  # Japanese
class XMINDRONProcessor(XMINDProcessor): ...  # Romanian
class XMINDSOMProcessor(XMINDProcessor): ...  # Somali
class XMINDSWHProcessor(XMINDProcessor): ...  # Swahili
class XMINDTAMProcessor(XMINDProcessor): ...  # Tamil
class XMINDTHAProcessor(XMINDProcessor): ...  # Thai
class XMINDTURProcessor(XMINDProcessor): ...  # Turkish
class XMINDVIEProcessor(XMINDProcessor): ...  # Vietnamese
