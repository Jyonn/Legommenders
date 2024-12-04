from loader.column_map import ColumnMap
from loader.data_hub import DataHub
from loader.embedding.embedding_hub import EmbeddingHub
from loader.status import Status
from model.meta_config import LegommenderMeta, LegommenderConfig


class Preparer:
    def __init__(
            self,
            meta: LegommenderMeta,
            status: Status,
            config: LegommenderConfig,
            column_map: ColumnMap,
            embedding_manager: EmbeddingHub,
            user_hub: DataHub,
            item_hub: DataHub,
    ):
        self.meta = meta
        self.status = status

        self.config = config

        self.embedding_manager = embedding_manager

        self.user_hub = user_hub
        self.item_hub = item_hub

        self.column_map = column_map  # type: ColumnMap
