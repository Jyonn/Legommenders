"""
manager.py

One-stop convenience wrapper that ties together *everything* required to
run an experiment with **Legommender**:

    • loads / prepares all UnitTypes (items, users, interactions)
    • selects the concrete *operator / predictor* classes from
      `ClassHub` according to the yaml-config
    • builds a fully fledged `LegoConfig`   (operators, predictor, …)
    • instantiates
          – `Legommender` (the neural network)
          – `Resampler`   (dataset pre-processing helper)
    • creates the individual `DataSet` objects for
          train / dev / test / fast-eval
    • hands out ready-to-use `DataLoader`s
    • switches `loader.env.Env` flags when switching between
      train / dev / test (so that the model behaves accordingly)
    • performs negative-sample filtering on the interaction UT when the
      corresponding flag is enabled

Despite its length **Manager** only contains *glue* code – it does *not*
implement any ML logic on its own.

Public interface
----------------
manager = Manager(cfg_data, cfg_embed, cfg_model, cfg_exp)

loader_train = manager.get_train_loader()
loader_dev   = manager.get_dev_loader()
loader_test  = manager.get_test_loader()

The instance also exposes a bunch of shortcuts such as `.train_ut`,
`.item_ut`, `.lego_config`, `.legommender`, … which are useful during
debugging.

Implementation notes
--------------------
1) All heavy-weight objects (datasets, models, embedding hub) are built
   exactly *once* inside the constructor – `Manager` is intended to live
   for the whole lifetime of the experiment script.

2) The class carefully respects the **negative-sampling** and
   **fast-evaluation** modes requested by the yaml:
        • `_negative_sampling()` removes positive samples from the dev
          set when the user opted for x-entropy training.
        • `create_fast_ut()` creates an artificial interaction UT that
          contains one dummy candidate per user (needed by FastItemPager
          and similar helpers).

3) A global `Env` object is used to signal the current phase
   (train / dev / test) to many sub-modules – `Manager.setup()` is the
   single place where these flags are set.

Only comments / doc-strings have been added – **runtime behaviour is
unchanged**.
"""

from __future__ import annotations

from typing import Type, Optional

from pigmento import pnt
from torch.utils.data import DataLoader
from unitok import Symbol

from loader.class_hub import ClassHub
from loader.column_map import ColumnMap
from loader.data_set import DataSet
from loader.embedding_hub import EmbeddingHub
from loader.env import Env
from loader.resampler import Resampler
from loader.symbols import Symbols
from loader.ut.lego_ut import LegoUT
from loader.ut.ut_hub import UTHub

from model.lego_config import LegoConfig
from model.legommender import Legommender
from model.operators.base_operator import BaseOperator
from model.predictors.base_predictor import BasePredictor
from utils.structure import Structure


class Manager:
    """
    Orchestrates data preparation, model construction and loader
    creation for one experimental run.

    The constructor performs *all* heavy lifting; afterwards only cheap
    bookkeeping is needed when switching phases.
    """

    # ------------------------------------------------------------------ #
    # *Class-level* attributes are only used for type annotations        #
    # ------------------------------------------------------------------ #
    item_ut: LegoUT
    user_ut: LegoUT
    fast_ut: LegoUT                         # synthetic UT for fast eval
    item_inputs: list[str]

    inter_uts: dict[Symbol, LegoUT]
    inter_sets: dict[Symbol, DataSet]

    cm: ColumnMap

    item_operator_class: Optional[Type[BaseOperator]]
    user_operator_class: Type[BaseOperator]
    predictor_class: Type[BasePredictor]

    lego_config: LegoConfig
    legommender: Legommender
    resampler: Resampler

    modes = {Symbols.train, Symbols.dev, Symbols.test}

    eh: EmbeddingHub

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self, data, embed, model, exp):
        """
        Parameters correspond to the four yaml sections:
            • data
            • embed
            • model
            • exp
        The concrete structure is already validated by the surrounding
        config loader – `Manager` just calls the `.xyz()` helpers.
        """
        self.data = data
        self.embed = embed
        self.model = model
        self.exp = exp

        # -------------------------------------------------------------- #
        # Sequential build-up                                            #
        # -------------------------------------------------------------- #
        self.load_data()            # → UnitTypes, ColumnMap …
        self.load_model_configs()   # → item/user op class, predictor …

        self.load_embeddings()      # → EmbeddingHub

        # Build operators / predictor (→ inside LegoConfig)
        self.lego_config.build_components()
        self.lego_config.register_inputer_vocabs()

        # Instantiate the actual network & helpers
        self.legommender = Legommender(self.lego_config)
        self.resampler = Resampler(self.lego_config)

        # Wrap UTs with DataSet + Resampler
        self.load_datasets()

    # ------------------------------------------------------------------ #
    # 1) Dataset / UnitType handling                                     #
    # ------------------------------------------------------------------ #
    def load_items(self):
        """
        Prepare the *item* UnitType – may include replication of
        columns and truncation masks as requested by the yaml.
        """
        ut = UTHub.get(self.data.item.ut)
        inputs = self.data.item.inputs()

        input_cols = []
        selected_attrs = {}

        for value in inputs:
            if isinstance(value, dict):  # {"col": truncate}
                col, truncate = next(iter(value.items()))
            else:
                col, truncate = value, None

            with ut:
                current_col = None
                if "->" in col:          # deep replicate
                    current_col, col = map(str.strip, col.split("->"))
                    ut.replicate(current_col, col, lazy=False)
                elif "-->" in col:       # shallow replicate
                    current_col, col = map(str.strip, col.split("-->"))
                    ut.replicate(current_col, col, lazy=True)
                if current_col is not None:
                    pnt(f"Replicated {current_col} to {col}")

            selected_attrs[col] = truncate
            input_cols.append(col)

        # Make sure the key feature is always present
        selected_attrs[ut.key_feature.name] = None
        ut.rebuild(selected_attrs)
        return ut, input_cols

    def load_users(self):
        ut = UTHub.get(self.data.user.ut)
        if self.data.user.truncate:
            ut.retruncate(
                feature=ut.meta.features[self.cm.history_col],
                truncate=self.data.user.truncate,
            )
        return ut

    def load_interactions(self):
        uts = {}
        for mode in self.modes:
            uts[mode] = UTHub.get(self.data.inter[mode.name], use_filter_cache=True)
        return uts

    def create_fast_ut(self):
        """
        Build a ‘dummy’ interaction UT that contains exactly *one*
        candidate per user.  This greatly speeds up the caching step
        during evaluation because we can iterate *user-wise* instead of
        *interaction-wise*.
        """
        ut: LegoUT = LegoUT.load(self.data.inter.test)
        user_num = len(self.user_ut)
        reset_data = {
            ut.key_feature.name: list(range(user_num)),
            self.cm.item_col: [[0] for _ in range(user_num)],
            self.cm.user_col: list(range(user_num)),
            self.cm.label_col: [[0] for _ in range(user_num)],
            # group col might coincide with user col
            self.cm.group_col: list(range(user_num)),
        }
        ut.reset(reset_data)
        return ut

    def load_data(self):
        """
        Collect all UnitTypes and apply column-map & optional filters.
        """
        # Column map is global for the whole project
        self.cm = ColumnMap(**self.data.column_map())

        # Item / User UTs
        self.item_ut, self.item_inputs = self.load_items()
        self.user_ut = self.load_users()

        # Interaction UTs (train / dev / test)
        self.inter_uts = self.load_interactions()

        # Synthetic UT for fast evaluation
        self.fast_ut = self.create_fast_ut()
        self.inter_uts[Symbols.fast_eval] = self.fast_ut

        # Make sure all interaction UTs contain *all* user columns
        for mode in self.inter_uts:
            with self.inter_uts[mode] as ut:
                ut.union(self.user_ut, soft_union=False)

        # Optional filters defined in yaml
        if self.data.inter.filters:
            for col in self.data.inter.filters:
                for filter_str in self.data.inter.filters[col]:
                    for mode in self.inter_uts:
                        ut = self.inter_uts[mode]
                        before = len(ut)
                        ut.filter(filter_str, col=col)
                        pnt(
                            f"Filter {col} with {filter_str} in {mode} phase: "
                            f"{before} → {len(ut)}"
                        )

        # Final step: teach ColumnMap all vocab columns
        self.cm.set_column_vocab(self.fast_ut)

    # ------------------------------------------------------------------ #
    # 2) Model / operator config                                         #
    # ------------------------------------------------------------------ #
    def load_model_configs(self):
        """
        Retrieve the *classes* of item/user operators and predictor,
        instantiate a `LegoConfig`, and (optionally) perform basic
        negative-sample filtering.
        """
        operators = ClassHub.operators()
        predictors = ClassHub.predictors()

        # Item operator may be disabled
        self.item_operator_class = (
            operators(self.model.meta.item) if self.model.meta.item else None
        )
        self.user_operator_class = operators(self.model.meta.user)
        self.predictor_class = predictors(self.model.meta.predictor)

        pnt(
            f"Selected Item Encoder : "
            f"{self.item_operator_class.__name__ if self.item_operator_class else 'null'}"
        )
        pnt(f"Selected User Encoder : {self.user_operator_class.__name__}")
        pnt(f"Selected Predictor     : {self.predictor_class.__name__}")

        self.lego_config = LegoConfig(**self.model.config())
        self.lego_config.set_component_classes(
            item_operator_class=self.item_operator_class,
            user_operator_class=self.user_operator_class,
            predictor_class=self.predictor_class,
        )
        self.lego_config.set_item_ut(self.item_ut, self.item_inputs)
        self.lego_config.set_user_ut(self.user_ut, [self.cm.history_col])
        self.lego_config.set_column_map(self.cm)

        if self.lego_config.use_neg_sampling:
            self.negative_sampling()

    # ------------------------------------------------------------------ #
    # 3) Embedding hub                                                   #
    # ------------------------------------------------------------------ #
    def load_embeddings(self):
        self.eh = EmbeddingHub(
            embedding_dim=self.lego_config.item_hidden_size,
            transformation=self.embed.transformation,
            transformation_dropout=self.embed.transformation_dropout,
        )
        # Pre-trained embeddings
        for info in self.embed.embeddings:
            self.eh.load_pretrained_embedding(**info())

        # Register vocabs that will be needed later
        if self.lego_config.use_item_content:
            self.eh.register_ut(self.item_ut, self.item_inputs)
        else:
            self.eh.register_vocab(self.item_ut.key_feature.tokenizer.vocab)

        self.lego_config.set_embedding_hub(self.eh)

    # ------------------------------------------------------------------ #
    # 4) Optional negative-sampling filter                               #
    # ------------------------------------------------------------------ #
    def negative_sampling(self):
        modes = [Symbols.train]
        if self.exp.policy.simple_dev:
            modes.append(Symbols.dev)

        filter_str = "lambda x: x == 1"

        for mode in modes:
            ut = self.inter_uts[mode]
            if not ut:
                continue
            before = len(ut)
            ut.filter(filter_str, col=self.cm.label_col)
            pnt(
                f"Filter {self.cm.label_col} with {filter_str} in {mode} phase: "
                f"{before} → {len(ut)}"
            )

    # ------------------------------------------------------------------ #
    # 5) Wrap UTs in DataSet + Resampler                                 #
    # ------------------------------------------------------------------ #
    def load_datasets(self):
        self.inter_sets = {
            mode: DataSet(ut, resampler=self.resampler)
            for mode, ut in self.inter_uts.items()
        }

    # ------------------------------------------------------------------ #
    # Pretty printing helpers                                            #
    # ------------------------------------------------------------------ #
    def stringify(self):
        for mode in self.modes:
            if mode in self.inter_uts:
                pnt(self.inter_uts[mode][0])
                break
        for mode in self.modes:
            if mode in self.inter_uts:
                pnt(Structure().analyse_and_stringify(self.inter_sets[mode][0]))
                break

    # ------------------------------------------------------------------ #
    # DataLoader helpers                                                 #
    # ------------------------------------------------------------------ #
    def _get_loader(self, mode: Symbol):
        return DataLoader(
            dataset=self.inter_sets[mode],
            shuffle=mode is Symbols.train,
            batch_size=self.exp.policy.batch_size,
            pin_memory=self.exp.policy.pin_memory,
            num_workers=5,
        )

    # Shortcuts
    train_ut = property(lambda self: self.inter_uts[Symbols.train])
    dev_ut = property(lambda self: self.inter_uts[Symbols.dev])
    test_ut = property(lambda self: self.inter_uts[Symbols.test])

    train_set = property(lambda self: self.inter_sets[Symbols.train])
    dev_set = property(lambda self: self.inter_sets[Symbols.dev])
    test_set = property(lambda self: self.inter_sets[Symbols.test])

    # Public loaders
    def get_train_loader(self, setup=Symbols.train):
        self.setup(setup)
        return self._get_loader(Symbols.train)

    def get_dev_loader(self, setup=Symbols.dev):
        self.setup(setup)
        return self._get_loader(Symbols.dev)

    def get_test_loader(self, setup=Symbols.test):
        self.setup(setup)
        return self._get_loader(Symbols.test)

    # ------------------------------------------------------------------ #
    # Phase switching                                                    #
    # ------------------------------------------------------------------ #
    def setup(self, mode: Symbol):
        """
        Switch the global `Env` flags, set PyTorch modules to the
        correct state (train / eval) and handle *representation caches*.
        """
        if mode is Symbols.train:
            Env.train()
            self.legommender.train()
            self.legommender.cacher.clean()
            return

        # ─ evaluation / test ─
        if mode is Symbols.dev:
            Env.dev()
        else:  # Symbols.test
            Env.test()

        self.legommender.eval()

        # Build (or refresh) the item / user caches
        self.legommender.cacher.cache(
            item_contents=self.resampler.item_cache,
            user_contents=self.inter_sets[Symbols.fast_eval],
        )
