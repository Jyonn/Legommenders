import os.path

import numpy as np

from embedder.base_embedder import BaseEmbedder
from loader.class_hub import ClassHub
from utils.config_init import CommandInit

if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['model'],
    ).parse()

    model = configuration.model
    embedder_hub = ClassHub.embedders()

    if model not in embedder_hub:
        raise ValueError(f'Unknown model: {model}')

    embedder: BaseEmbedder = embedder_hub[model]()
    embeddings: np.ndarray = embedder.get_embeddings()

    export_dir = os.path.join('data', 'embeddings')
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f'{model}.npy')
    np.save(path, embeddings)

    print(f'Embeddings ({embeddings.shape}) saved to {path}')
