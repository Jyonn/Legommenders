"""
lego_config.py   (annotated version)

Central configuration class that *orchestrates* all components
(operators, predictor, datasets …) required by **Legommender**.  
It collects the user-specified hyper-parameters, creates the concrete
instances of the sub-modules and makes them accessible via public
attributes.

The class is intentionally *state-ful* – after calling
    >>> config = LegoConfig(...)
    >>> config.set_*()
    >>> config.build_components()
all relevant objects are stored inside `config` and can be reached by
other parts of the framework.

Overview of the most important responsibilities
-----------------------------------------------
1)  Hold global flags that influence the pipeline, e.g.
        • use_neg_sampling
        • use_item_content
        • cache_page_size
2)  Store references to *component classes* (set via
        `set_component_classes`) so that the concrete instances can be
        created on demand.
3)  Provide convenience setters for
        • column map            (`set_column_map`)
        • embedding hub          (`set_embedding_hub`)
        • item / user UT + inputs
4)  Instantiate
        • item operator  (optional)
        • user operator
        • predictor
    inside `build_components()`.
5)  Collect all *trainable* vocabularies from the inputers and register
    them in the shared `EmbeddingHub` (`register_inputer_vocabs`).

Note
----
The heavy lifting (e.g. neural network definitions) is *not* done in
this class – it merely wires together already implemented building
blocks.

"""

from __future__ import annotations
from typing import Type, Optional

from loader.column_map import ColumnMap
from loader.embedding_hub import EmbeddingHub
from loader.ut.lego_ut import LegoUT
from model.operators.base_operator import BaseOperator
from model.predictors.base_predictor import BasePredictor
from utils.function import combine_config


class LegoConfig:
    # ------------------------------------------------------------------ #
    # Class-level type annotations                                       #
    # ------------------------------------------------------------------ #
    cm: ColumnMap                       # mapping of dataframe columns
    eh: EmbeddingHub                    # global embedding store

    item_ut: LegoUT                     # unit-types for *items*
    item_inputs: list                   # feature list for item inputer
    user_ut: LegoUT                     # unit-types for *users*
    user_inputs: list                   # feature list for user inputer

    # Component classes (set by `set_component_classes`)
    item_operator_class: Type[BaseOperator]
    user_operator_class: Type[BaseOperator]
    predictor_class: Type[BasePredictor]

    # Instantiated components (after `build_components`)
    item_operator: Optional[BaseOperator]
    user_operator: BaseOperator
    predictor: BasePredictor

    # ------------------------------------------------------------------ #
    # Constructor – only stores raw hyper-parameters                     #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        hidden_size: int,
        user_config: dict,
        *,
        neg_count: int = 4,
        item_hidden_size: Optional[int] = None,
        item_config: Optional[dict] = None,
        predictor_config: Optional[dict] = None,
        use_neg_sampling: bool = True,
        use_item_content: bool = True,
        use_fast_eval: bool = True,
        item_page_size: int = 0,
        cache_page_size: int = 512,
        **kwargs,
    ):
        # -------- global network dimensions / flags -------- #
        self.hidden_size = hidden_size
        self.item_hidden_size = item_hidden_size or hidden_size
        self.use_item_content = use_item_content

        # -------- configs for sub-modules (may be None) ---- #
        # they will be merged with default settings later
        self.item_config = item_config
        self.user_config = user_config
        self.predictor_config = predictor_config or {}

        # -------- negative sampling ------------------------ #
        self.use_neg_sampling = use_neg_sampling
        self.neg_count = neg_count

        # -------- paging / caching behaviour --------------- #
        self.item_page_size = item_page_size
        self.cache_page_size = cache_page_size
        self.use_fast_eval = use_fast_eval

        # If item content is used but no config was provided,
        # create an empty dict to ease further merges.
        if self.use_item_content:
            self.item_config = self.item_config or {}

    # ------------------------------------------------------------------ #
    # Convenience setters                                                 #
    # ------------------------------------------------------------------ #
    def set_component_classes(
        self,
        item_operator_class: Type[BaseOperator],
        user_operator_class: Type[BaseOperator],
        predictor_class: Type[BasePredictor],
    ):
        """Register the concrete *classes* that should be instantiated
        later in `build_components()`."""
        self.item_operator_class = item_operator_class
        self.user_operator_class = user_operator_class
        self.predictor_class = predictor_class

    def set_item_ut(self, item_ut: LegoUT, item_inputs: list):
        """Provide the UnitType and feature definitions of *items*."""
        self.item_ut = item_ut
        self.item_inputs = item_inputs

    def set_user_ut(self, user_ut: LegoUT, user_inputs: list):
        """Provide the UnitType and feature definitions of *users*."""
        self.user_ut = user_ut
        self.user_inputs = user_inputs

    def set_column_map(self, cm: ColumnMap):
        """Connect the global `ColumnMap` (dataset column aliases)."""
        self.cm = cm

    def set_embedding_hub(self, eh: EmbeddingHub):
        """Attach the shared `EmbeddingHub` instance."""
        self.eh = eh

    # ------------------------------------------------------------------ #
    # Component construction                                              #
    # ------------------------------------------------------------------ #
    def build_components(self):
        """
        Instantiate item/user operators and predictor according to the
        previously registered classes / configs.

        The resulting objects are stored in
            • self.item_operator  (may be None)
            • self.user_operator
            • self.predictor
        """
        self.item_operator = None  # reset in case of re-builds

        # -------------- Item operator (optional) -------------- #
        if self.use_item_content:
            # Merge user config with mandatory arguments
            item_config = self.item_operator_class.config_class(
                **combine_config(
                    config=self.item_config,
                    hidden_size=self.hidden_size,
                    input_dim=self.item_hidden_size,
                )
            )

            self.item_operator = self.item_operator_class(
                config=item_config,
                target_user=False,
                lego_config=self,
            )

        # -------------- User operator (always present) -------- #
        # Determine input dimension: item operator output OR item hidden size
        user_input_dim = (
            self.item_operator.output_dim
            if self.use_item_content
            else self.item_hidden_size
        )

        user_config = self.user_operator_class.config_class(
            **combine_config(
                config=self.user_config,
                hidden_size=self.hidden_size,
                input_dim=user_input_dim,
            )
        )

        # Special case: *flatten mode* needs item UT/in-puts
        if self.user_operator_class.flatten_mode:
            user_config.inputer_config["item_ut"] = self.item_ut
            user_config.inputer_config["item_inputs"] = self.item_inputs

        self.user_operator = self.user_operator_class(
            config=user_config,
            target_user=True,
            lego_config=self,
        )

        # -------------- Predictor ------------------------------------- #
        # Sanity checks regarding negative sampling support
        if self.use_neg_sampling and not self.predictor_class.allow_matching:
            raise ValueError(
                f"{self.predictor_class.__name__} does not support negative sampling"
            )
        if not self.use_neg_sampling and not self.predictor_class.allow_ranking:
            raise ValueError(
                f"{self.predictor_class.__name__} only supports negative sampling"
            )

        predictor_config = self.predictor_class.config_class(
            **combine_config(
                config=self.predictor_config,
                hidden_size=self.hidden_size,
            )
        )

        self.predictor = self.predictor_class(
            config=predictor_config,
            lego_config=self,
        )

    # ------------------------------------------------------------------ #
    # Embedding-hub registration                                         #
    # ------------------------------------------------------------------ #
    def register_inputer_vocabs(self):
        """
        Collect the vocabularies required by the inputers of the
        instantiated operators and register them in the global
        `EmbeddingHub` (`self.eh`).

        This method must be called *after* `build_components()`.
        """
        # ----- item vocabularies (optional) -----
        if self.use_item_content:
            for vocab in self.item_operator.inputer.get_vocabs():
                self.eh.register_vocab(vocab)

        # ----- user vocabularies (always present) -----
        for vocab in self.user_operator.inputer.get_vocabs():
            self.eh.register_vocab(vocab)
