import os
import zipfile
import subprocess

import torch
from pigmento import pnt
from torch import nn
from unitok import Vocab

from embedder.base_embedder import BaseEmbedder


class GloVeEmbedder(BaseEmbedder):
    DL_URI = 'https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip'
    FILE_DIR = os.path.join('data', 'embeddings')
    ZIP_NAME = 'glove.6B.zip'
    ZIP_PATH = os.path.join(FILE_DIR, ZIP_NAME)
    FILE_NAME = 'glove.6B.300d.txt'
    FILE_PATH = os.path.join(FILE_DIR, FILE_NAME)

    def __init__(self):
        super().__init__()
        os.makedirs(self.FILE_DIR, exist_ok=True)

    def _get_embeddings(self):
        if not os.path.exists(self.ZIP_PATH):
            """Download GloVe embeddings from the specified URL."""
            pnt(f"Downloading GloVe embeddings from {self.DL_URI}...")
            subprocess.run(["curl", "-o", self.ZIP_PATH, self.DL_URI], check=True)
            print(f"GloVe downloaded to {self.ZIP_PATH}")

        if not os.path.exists(self.FILE_PATH):
            """Unzip the GloVe file."""
            print(f"Unzipping GloVe embeddings...")
            with zipfile.ZipFile(self.ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.FILE_DIR)
            print(f"Unzipped to {self.FILE_DIR}")

        """Extract unique tokens (words) from the GloVe file."""
        vocab = Vocab(name='glove')
        embeddings = []
        with open(self.FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                word, *vector = line.split()
                vocab.append(word)
                vector = [float(x) for x in vector]
                embeddings.append(vector)

        embeddings = torch.tensor(embeddings)
        vocab.save(self.FILE_DIR)
        pnt(f"Saved GloVe embeddings to {self.FILE_DIR}, "
            f"along with the vocabulary file {vocab.filepath(self.FILE_DIR)}")

        return nn.Embedding.from_pretrained(embeddings, freeze=True)
