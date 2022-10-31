from loader.depot.vocab_loader import VocabLoader
from loader.embedding.embedding_init import EmbeddingInit
from set.base_dataset import BaseDataset
from task.base_batch import HSeqBatch
from task.base_hseq_task import BaseHSeqTask


class RankingTask(BaseHSeqTask):
    name = 'ranking'
    dynamic_loader = False

    def __init__(
            self,
            dataset: BaseDataset,
            doc_depot,

    ):
        super().__init__(dataset, doc_depot)

    def negative_sampling(self):
        return []

    def get_embeddings(
            self,
            batch: HSeqBatch,
            embedding_init: EmbeddingInit,
            vocab_loader: VocabLoader,
    ):
        clicks_embedding, candidates_embedding = super(RankingTask, self).get_embeddings(
            batch=batch,
            embedding_init=embedding_init,
            vocab_loader=vocab_loader,
        )  # [B, N, L, D], [B, 1, L, D]
        input_embedding = self._get_embedding(
            inputs=batch.inputs,
            embedding_init=embedding_init,
            vocab_loader=vocab_loader,
        )  # [B, ]
        return input_embedding, clicks_embedding, candidates_embedding


