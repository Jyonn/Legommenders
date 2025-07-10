import os.path

import numpy as np
import yaml

from embedder.base_embedder import BaseEmbedder
from loader.class_hub import ClassHub
from utils import io
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

    print(f'Embeddings {embeddings.shape} saved to {path}')

    embed_config = dict(
        name=model,
        transformation='auto',
        transformation_dropout=0.1,
        embeddings=[
            dict(
                vocab_name='<vocab_name>',
                path=path,
                frozen=True,
            )
        ]
    )

    config_path = os.path.join('config', 'embed', f'{model}.yaml')

    # with open(config_path, 'w') as f:
    #     yaml.dump(embed_config, f)
    io.yaml_save(embed_config, config_path)

    print(f'Embedding configuration saved to {config_path}, please specify the vocab_name before using it')
